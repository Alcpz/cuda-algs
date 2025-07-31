#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

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
void clanker_gemm(int M, int N, int K, float alpha,
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
  clanker_gemm(M, N, K, 1.0, a_vec.data(), b_vec.data(), 0.0, c_ref_vec.data());
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

// TODO: helper functions for indexing logic, since we have linear
// TODO: Grid, Block dimensions

// Row Major matrices.
// Each Thread consumes K to provide a single value of C
// Expect M x N Threads

// Faster dimension is rows?
// A = M * K (Thread IDX is our M dim coord)
// B = K * N (Thread IDY is our N dim coord)
// B is NOT transposed
__global__ void naive_gemm(float *a_data, float *b_data, float *c_data, int M,
                           int N, int K) {
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
    our_c_value += a_data[idy * K + i] * b_data[i * N + idx];

  c_data[c_idx] = our_c_value;
}

int main() {
  constexpr int M = 1024;
  constexpr int N = 512;
  constexpr int K = 128;

  constexpr int a_size = M * K;
  constexpr int b_size = N * K;
  constexpr int c_size = M * N;

  std::vector<float> a_vec(a_size);
  std::vector<float> b_vec(b_size);
  std::vector<float> c_vec(c_size);
  std::vector<float> c_ref_vec(c_size);

  std::mt19937 rng;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::generate(a_vec.begin(), a_vec.end(), [&]() { return dist(rng); });
  std::generate(b_vec.begin(), b_vec.end(), [&]() { return dist(rng); });

  // std::iota(a_vec.begin(), a_vec.end(), 0.f);
  // std::iota(b_vec.begin(), b_vec.end(), 1.f);

  float *a_data;
  float *b_data;
  float *c_data;
  float *c_ref;

  cudaMalloc(reinterpret_cast<void **>(&a_data), sizeof(float) * a_size);
  cudaMalloc(reinterpret_cast<void **>(&b_data), sizeof(float) * b_size);
  cudaMalloc(reinterpret_cast<void **>(&c_data), sizeof(float) * c_size);
  cudaMalloc(reinterpret_cast<void **>(&c_ref), sizeof(float) * c_size);

  cudaMemcpy(a_data, a_vec.data(), sizeof(float) * a_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(b_data, b_vec.data(), sizeof(float) * b_size,
             cudaMemcpyHostToDevice);
  cudaMemset(c_data, 0, sizeof(float) * c_size);

  dim3 threads(16, 16);
  dim3 blocks(safe_div(N, threads.x), safe_div(M, threads.y));

  std::cout << "Blocks: " << blocks.x << " " << blocks.y << std::endl;
  std::cout << "Threads: " << threads.x << " " << threads.y << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  naive_gemm<<<blocks, threads>>>(a_data, b_data, c_data, M, N, K);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Error kernel " << err << std::endl;
  }
  cudaMemcpy(c_vec.data(), c_data, sizeof(float) * c_size,
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  print_flops("\nNaive Gemm", M, N, K, start, stop);
  verify(M, N, K, a_vec, b_vec, c_vec, c_ref_vec);

  // Verify
  cudaFree(a_data);
  cudaFree(b_data);
  cudaFree(c_data);
  cudaFree(c_ref);
  return 0;
}
