#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#define CUDA_CHECK(ans)                                                        \
  do {                                                                         \
    cudaError_t err__ = (ans);                                                 \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),   \
              __FILE__, __LINE__);                                             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

using GemmKernel = void (*)(const float *, const float *, float *, int, int,
                            int);

struct KernelConfig {
  std::string name;
  GemmKernel kernel;
  dim3 grid;
  dim3 block;
};

static inline float elapsed_ms(cudaEvent_t s, cudaEvent_t e) {
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
  return ms;
}

int safe_div(int a, int b) { return (a + b - 1) / b; }

bool isAlmostEqual(float val, float ref, float eps = 1.0e-3) {
  return (std::abs(val - ref) < (std::abs(val) * eps));
}

void print_flops(std::string fname, int M, int N, int K, cudaEvent_t start,
                 cudaEvent_t stop) {
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  double flops = 2.0 * M * N * K;
  double gflops = (flops / (ms / 1000.0)) / 1e9;
  std::cout << fname << ": " << ms << " ms, " << gflops << " Gflops\n";
}

// Reference GEMM from chatgpt (our good friend & respected colleague)
void ref_gemm(int M, int N, int K, float alpha,
              const float *A, // M x K
              const float *B, // K x N
              float beta,
              float *C) // M x N
{
  for (int i = 0; i < M; ++i) {   // rows of A and C
    for (int j = 0; j < N; ++j) { // columns of B and C
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {         // columns of A, rows of B
        sum += A[i * K + k] * B[k * N + j]; // row-major indexing
      }
      C[i * N + j] = alpha * sum + beta * C[i * N + j];
    }
  }
}

template <typename T>
void verify(int M, int N, int K, std::vector<T> a_vec, std::vector<T> b_vec,
            std::vector<T> c_vec, std::vector<T> c_ref_vec, int limit = 25) {
  ref_gemm(M, N, K, 1.0, a_vec.data(), b_vec.data(), 0.0, c_ref_vec.data());
  int wrong = 0;
  for (int i = 0; i < M * N; ++i) {
    if (!isAlmostEqual(c_ref_vec[i], c_vec[i])) {
      std::cout << "You are terrible at GEMMS " << i << " C: " << c_vec[i]
                << " Ref: " << c_ref_vec[i]
                << " diff: " << std::abs(c_vec[i] - c_ref_vec[i]) << std::endl;
      if (++wrong == limit) {
        std::cout << "Error limit (" << limit << ") reached: EXITING"
                  << std::endl;
        break;
      }
    }
  }
}

void run_gemm_suite(const std::vector<KernelConfig> &ks, const float *a_data,
                    const float *b_data, float *c_data, int M, int N, int K,
                    const std::vector<float> &a_host,
                    const std::vector<float> &b_host,
                    std::vector<float> &c_host) {
  std::vector<float> c_ref_vec(M * N);
  for (const auto &cfg : ks) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    cfg.kernel<<<cfg.grid, cfg.block>>>(a_data, b_data, c_data, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = elapsed_ms(start, stop);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpyAsync(c_host.data(), c_data,
                               sizeof(float) * size_t(M) * N,
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Report GFLOP/s (2*M*N*K flops)
    const double flops = 2.0 * M * N * K;
    const double gflops = (flops / (ms / 1000.0)) / 1e9;
    printf("%-20s (%2d, %2d) (%2d, %2d) %8.3f ms   %8.2f GFLOP/s\n",
           cfg.name.c_str(), cfg.grid.x, cfg.grid.y, cfg.block.x, cfg.block.y,
           ms, gflops);

    verify(M, N, K, a_host, b_host, c_host, c_ref_vec);
  }
}

// TODO: helper functions for indexing logic, since we have linear
// TODO: Grid, Block dimensions

// Row Major matrices.
// Each Thread consumes K to provide a single value of C
// Expect M x N Threads

// A = M * K (Thread IDX is our M dim coord)
// B = K * N (Thread IDY is our N dim coord)
// B is NOT transposed
__global__ void naive_gemm(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           int M, int N, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  // 2d threads
  // Each thread loops over K dim
  // AKA each thread is responsible of a single value in C
  // Our row-major C matrix is N-major
  // Define idx along the row for coalesced writes of C
  int c_idx = N * idy + idx;
  float our_c_value = 0.0;
  for (int i = 0; i < K; i++)
    our_c_value += A[idy * K + i] * B[i * N + idx];

  C[c_idx] = our_c_value;
}

constexpr int TILE_SIZE = 16;

__global__ void shmem_gemm(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           int M, int N, int K) {
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  // Load into shared mem
  __shared__ float sh_a[TILE_SIZE][TILE_SIZE];
  __shared__ float sh_b[TILE_SIZE][TILE_SIZE];

  float acc = 0.0f;
  for (int v = 0; v < K; v += TILE_SIZE) {
    int tile_a_col = threadIdx.x + v;
    int tile_b_row = threadIdx.y + v;
    if (row < M && tile_a_col < K)
      sh_a[threadIdx.y][threadIdx.x] = A[row * K + tile_a_col];
    else
      sh_a[threadIdx.y][threadIdx.x] = 0.0f;
    if (col < N && tile_b_row < K)
      sh_b[threadIdx.y][threadIdx.x] = B[tile_b_row * N + col];
    else
      sh_b[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();

#pragma unroll
    for (int n = 0; n < TILE_SIZE; n++) {
      acc += sh_a[threadIdx.y][n] * sh_b[n][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = acc;
}

__global__ void buffered_shmem_gemm(const float *__restrict__ A,
                                    const float *__restrict__ B,
                                    float *__restrict__ C, int M, int N,
                                    int K) {
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  // Load into shared mem
  __shared__ float sh_a[2][TILE_SIZE][TILE_SIZE];
  __shared__ float sh_b[2][TILE_SIZE][TILE_SIZE];

  float acc = 0.0f;
  // Load A and B
  int tile_a_col = threadIdx.x;
  int tile_b_row = threadIdx.y;
  if (row < M && tile_a_col < K)
    sh_a[0][threadIdx.y][threadIdx.x] = A[row * K + threadIdx.x];
  else
    sh_a[0][threadIdx.y][threadIdx.x] = 0.0f;
  if (col < N && tile_b_row < K)
    sh_b[0][threadIdx.y][threadIdx.x] = B[threadIdx.y * N + col];
  else
    sh_b[0][threadIdx.y][threadIdx.x] = 0.0f;
  __syncthreads();

  int cur = 0;
  for (int v = 0; v < K; v += TILE_SIZE) {
    int next = cur ^ 1;
    if (v + TILE_SIZE < K) {
      tile_a_col = TILE_SIZE + v + threadIdx.x;
      tile_b_row = TILE_SIZE + v + threadIdx.y;
      // Prefetch A
      if (row < M && tile_a_col < K)
        sh_a[next][threadIdx.y][threadIdx.x] = A[row * K + tile_a_col];
      else
        sh_a[next][threadIdx.y][threadIdx.x] = 0.0f;
      if (col < N && tile_b_row < K)
        sh_b[next][threadIdx.y][threadIdx.x] = B[tile_b_row * N + col];
      else
        sh_b[next][threadIdx.y][threadIdx.x] = 0.0f;
    }

#pragma unroll
    for (int n = 0; n < TILE_SIZE; n++) {
      acc += sh_a[cur][threadIdx.y][n] * sh_b[cur][n][threadIdx.x];
    }
    cur = next;
    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = acc;
}

constexpr int TILE_SIZE_REG = 32;
constexpr int RM = 4;
__global__ void registers_1d_gemm(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C, int M, int N, int K) {

  int base_row = TILE_SIZE_REG * blockIdx.y;
  int base_col = TILE_SIZE_REG * blockIdx.x;

  int col = threadIdx.x + base_col;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  __shared__ float sh_a[2][TILE_SIZE_REG][TILE_SIZE_REG];
  __shared__ float sh_b[2][TILE_SIZE_REG][TILE_SIZE_REG];
  float acc[RM] = {0.0f};

  // Load A and B
  int tile_a_col = threadIdx.x;
  int tile_b_row = threadIdx.y;
#pragma unroll
  for (int j = 0; j < RM; ++j) {
    const int local_row = j * blockDim.y + threadIdx.y;
    const int global_row = base_row + local_row;

    if (global_row < M && tile_a_col < K)
      sh_a[0][local_row][tile_a_col] = A[global_row * K + tile_a_col];
    else
      sh_a[0][local_row][tile_a_col] = 0.f;
  }

  for (int j = 0; j < TILE_SIZE_REG; j += blockDim.y) {
    if (col < N && tile_b_row < K)
      sh_b[0][ty + j][tx] = B[(tile_b_row + j) * N + col];
    else
      sh_b[0][ty + j][tx] = 0.0f;
  }
  __syncthreads();

  int cur = 0;
  for (int v = 0; v < K; v += TILE_SIZE_REG) {
    int next = cur ^ 1;
    if (v + TILE_SIZE_REG < K) {
      tile_a_col = TILE_SIZE_REG + v + threadIdx.x;
      tile_b_row = TILE_SIZE_REG + v + threadIdx.y;

#pragma unroll
      for (int j = 0; j < RM; ++j) {
        const int local_row = j * blockDim.y + threadIdx.y;
        const int global_row = base_row + local_row;

        if (global_row < M && tile_a_col < K)
          sh_a[next][local_row][tx] = A[global_row * K + tile_a_col];
        else
          sh_a[next][local_row][tx] = 0.f;
      }

      for (int j = 0; j < TILE_SIZE_REG; j += blockDim.y) {
        if (col < N && tile_b_row < K)
          sh_b[next][ty + j][tx] = B[(tile_b_row + j) * N + col];
        else
          sh_b[next][ty + j][tx] = 0.0f;
      }
    }

#pragma unroll
    for (int k = 0; k < TILE_SIZE_REG; k++) {
#pragma unroll
      for (int j = 0; j < RM; j++) {
        const int local_row = j * blockDim.y + threadIdx.y;

        acc[j] += sh_a[cur][local_row][k] * sh_b[cur][k][tx];
      }
    }
    __syncthreads();
    cur = next;
  }

#pragma unroll
  for (int j = 0; j < RM; ++j) {
    const int local_row = j * blockDim.y + threadIdx.y;
    const int global_row = base_row + local_row;

    if (global_row < M && col < N) {
      C[global_row * N + col] = acc[j];
    }
  }
}

int main() {
  constexpr int M = 1024;
  constexpr int N = 512;
  constexpr int K = 128;

  // constexpr int M = 64;
  // constexpr int N = 64;
  // constexpr int K = 64;

  std::cout << "M: " << M << " ";
  std::cout << "N: " << N << " ";
  std::cout << "K: " << K << " ";
  std::cout << std::endl;

  constexpr int a_size = M * K;
  constexpr int b_size = N * K;
  constexpr int c_size = M * N;

  std::vector<float> a_vec(a_size);
  std::vector<float> b_vec(b_size);
  std::vector<float> c_vec(c_size);

  // std::mt19937 rng;
  // std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  // std::generate(a_vec.begin(), a_vec.end(), [&]() { return dist(rng); });
  // std::generate(b_vec.begin(), b_vec.end(), [&]() { return dist(rng); });

  // std::iota(a_vec.begin(), a_vec.end(), 0.f);
  // std::iota(b_vec.begin(), b_vec.end(), 1.f);

  std::fill(a_vec.begin(), a_vec.end(), 1);
  std::fill(b_vec.begin(), b_vec.end(), 1);

  float *a_data;
  float *b_data;
  float *c_data;

  cudaMalloc(reinterpret_cast<void **>(&a_data), sizeof(float) * a_size);
  cudaMalloc(reinterpret_cast<void **>(&b_data), sizeof(float) * b_size);
  cudaMalloc(reinterpret_cast<void **>(&c_data), sizeof(float) * c_size);

  cudaMemcpy(a_data, a_vec.data(), sizeof(float) * a_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(b_data, b_vec.data(), sizeof(float) * b_size,
             cudaMemcpyHostToDevice);

  dim3 threads(TILE_SIZE, TILE_SIZE);
  dim3 blocks(safe_div(N, threads.x), safe_div(M, threads.y));

  dim3 threads_reg(TILE_SIZE_REG, TILE_SIZE_REG / RM);
  dim3 blocks_reg(safe_div(N, threads_reg.x), safe_div(M, threads_reg.y * RM));

  std::vector<KernelConfig> suite = {
      KernelConfig{"Naive", naive_gemm, blocks, threads},
      KernelConfig{"Shmem", shmem_gemm, blocks, threads},
      KernelConfig{"Shmem (buffer)", buffered_shmem_gemm, blocks, threads},
      KernelConfig{"registers (1D)", registers_1d_gemm, blocks_reg,
                   threads_reg},
  };

  // Reuse host buffers you already have
  run_gemm_suite(suite, a_data, b_data, c_data, M, N, K, a_vec, b_vec, c_vec);

  // Verify
  cudaFree(a_data);
  cudaFree(b_data);
  cudaFree(c_data);
  return 0;
}
