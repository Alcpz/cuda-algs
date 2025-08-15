#include <cassert>
#include <cmath>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <vector>

constexpr float EPSILON = 1e-6;

constexpr int safe_div(int a, int b) { return (a + b - 1) / b; }

#define CUDA_CHECK(ans)                                                        \
  do {                                                                         \
    cudaError_t err__ = (ans);                                                 \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),   \
              __FILE__, __LINE__);                                             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

using LayerNormKernel = void (*)(const float *, float *, int, int);

struct KernelConfig {
  std::string name;
  LayerNormKernel kernel;
  dim3 grid;
  dim3 block;
  int shared_mem_bytes;
  int iterations;
};

static inline float elapsed_ms(cudaEvent_t s, cudaEvent_t e) {
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
  return ms;
}

bool isAlmostEqual(float val, float ref, float eps = 1.0e-5) {
  if (std::isnan(val) && std::isnan(ref))
    return true;
  if (std::isinf(val) || std::isinf(ref))
    return val == ref;

  // Standard: |a-b| <= atol + rtol * max(|a|, |b|)
  const float diff = std::fabs(val - ref);
  const float scale = std::max(std::fabs(val), std::fabs(ref));
  // printf("\nval:%.8f ref: %.8f, diff: %.8f tol: %.8f", val, ref, diff,
  //        eps + eps * scale);
  return diff <= (eps + eps * scale);
}

void ref_layernorm(const std::vector<float> &x, std::vector<float> &y,
                   std::size_t M, std::size_t N) {
  for (int i = 0; i < M; ++i) {
    const float *xi = &x[i * N];
    float *yi = &y[i * N];

    // mean
    float mean = 0.f;
    for (int j = 0; j < N; ++j)
      mean += xi[j];
    mean /= static_cast<float>(N);

    // variance
    float var = 0.f;
    for (int j = 0; j < N; ++j) {
      float d = xi[j] - mean;
      var += d * d;
    }
    var /= static_cast<float>(N);

    // normalize
    float rstd = 1.0f / std::sqrt(var + EPSILON);
    for (int j = 0; j < N; ++j)
      yi[j] = (xi[j] - mean) * rstd;
    // printf("%.2f %.2f %.2f\n", yi[0], yi[1], yi[2]);
  }
}

template <typename T>
void verify(std::vector<T> x, std::vector<T> y_vec, std::size_t M,
            std::size_t N, int limit = 25) {
  std::vector<float> y_ref_vec(M * N);
  ref_layernorm(x, y_ref_vec, M, N);
  int wrong = 0;
  for (int i = 0; i < M * N; ++i) {
    if (!isAlmostEqual(y_ref_vec[i], y_vec[i])) {
      std::cout << "You are terrible at LAYER NORM " << i << " C: " << y_vec[i]
                << " Ref: " << y_ref_vec[i]
                << " diff: " << std::fabs(y_vec[i] - y_ref_vec[i]) << std::endl;
      if (++wrong == limit) {
        std::cout << "Error limit (" << limit << ") reached: EXITING"
                  << std::endl;
        break;
      }
    }
  }
}

void run_layernorm_suite(const std::vector<KernelConfig> &ks, const float *x,
                         float *y, std::size_t M, std::size_t N,
                         const std::vector<float> &x_host,
                         std::vector<float> &y_host) {

  std::size_t size = M * N;
  for (const auto &cfg : ks) {
    printf("%-20s (%2d, %2d) (%2d, %2d)", cfg.name.c_str(), cfg.grid.x,
           cfg.grid.y, cfg.block.x, cfg.block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < cfg.iterations; i++)
      cfg.kernel<<<cfg.grid, cfg.block, cfg.shared_mem_bytes>>>(x, y, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = elapsed_ms(start, stop) / cfg.iterations;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpyAsync(y_host.data(), y,
                               sizeof(float) * std::size_t(M * N),
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    const double flops = M * (6 * N + 3);
    const double gflops = (flops / (ms / 1000.0)) / 1e9;
    const double mbytes = M * N * 4 / 1e6;
    const double mbytess = mbytes / (ms / 1000.0);
    printf("%8.3f ms   %8.2f GFLOP/s   %8.2f Mbytes/s\n", ms, gflops, mbytess);

    if (size <= 2048) {
      verify(x_host, y_host, M, N);
    }
  }

  if (size > 2048) {
    std::cout << "  Skipping Verification (size " << size << " > 2048)"
              << std::endl;
  }
}

__global__ void layernorm_naive(const float *__restrict__ x,
                                float *__restrict__ y, int M, int N) {
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  if (row >= M)
    return;

  const float *xi = &x[row * N];
  float *yi = &y[row * N];

  float mean = 0.0f;
  for (int j = 0; j < N; j++) {
    mean += xi[j];
  }
  mean /= N;

  float var = 0.0f;
  for (int j = 0; j < N; j++) {
    const float std_dif = xi[j] - mean;
    var += std_dif * std_dif;
  }
  var /= N;

  float stddev = sqrt(var + EPSILON);
  for (int j = 0; j < N; j++) {
    yi[j] = (xi[j] - mean) / stddev;
  }
}

inline __device__ float warp_reduce_bcast(float val) {
  float reg = val;
  reg += __shfl_down_sync(0xFFFFFFFF, reg, 16);
  reg += __shfl_down_sync(0xFFFFFFFF, reg, 8);
  reg += __shfl_down_sync(0xFFFFFFFF, reg, 4);
  reg += __shfl_down_sync(0xFFFFFFFF, reg, 2);
  reg += __shfl_down_sync(0xFFFFFFFF, reg, 1);
  reg = __shfl_sync(0xFFFFFFFF, reg, 0);
  return reg;
}

__global__ void layernorm_1pass(const float *__restrict__ x,
                                float *__restrict__ y, int M, int N) {
  int row = blockIdx.x;
  if (row >= M)
    return;

  extern __shared__ float shmem[];

  int tx = threadIdx.x;

  const float *xi = &x[row * N];
  float *yi = &y[row * N];

  float partial_mean = 0.0f;
  float partial_var = 0.0f;
  for (int i = tx; i < N; i += blockDim.x) {
    const float val = xi[i];
    partial_mean += val;
    partial_var += val * val;
  }
  shmem[tx] = partial_mean;
  __syncthreads();

  // Calcs mean
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tx + stride < N) {
      shmem[tx] += shmem[tx + stride];
    }
    __syncthreads();
  }
  float mean = shmem[0] / N;
  __syncthreads();

  shmem[tx] = partial_var;
  __syncthreads();
  // Calcs Variance
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tx + stride < N) {
      shmem[tx] += shmem[tx + stride];
    }
    __syncthreads();
  }
  float var = shmem[0] / N - (mean * mean);

  float stddev = sqrt(var + EPSILON);
  for (int j = 0; j < N; j++) {
    yi[j] = (xi[j] - mean) / stddev;
  }
}

__global__ void layernorm_warpreduce(const float *__restrict__ x,
                                     float *__restrict__ y, int M, int N) {
  int row = blockIdx.x;
  if (row >= M)
    return;

  int tx = threadIdx.x;

  const float *xi = &x[row * N];
  float *yi = &y[row * N];

  float partial_mean = 0.0f;
  float partial_var = 0.0f;
  for (int i = tx; i < N; i += blockDim.x) {
    const float val = xi[i];
    partial_mean += val;
    partial_var += val * val;
  }
  float inv = 1.0f / N;
  float mean = warp_reduce_bcast(partial_mean) * inv;
  float var = warp_reduce_bcast(partial_var) * inv - (mean * mean);

  float rstddev = rsqrtf(var + EPSILON);
  for (int j = tx; j < N; j += blockDim.x) {
    yi[j] = (xi[j] - mean) * rstddev;
  }
}

__global__ void layernorm_vectorize(const float *__restrict__ x,
                                    float *__restrict__ y, int M, int N) {
  int row = blockIdx.x;
  if (row >= M)
    return;

  int tx = threadIdx.x;
  int values = N / 4;

  const float *xi = &x[row * N];
  const float4 *xi4 = reinterpret_cast<const float4 *>(xi);

  float partial_mean = 0.0f;
  float partial_var = 0.0f;

#pragma unroll
  for (int i = tx; i < values; i += 32) {
    const float4 val = xi4[i];
    float sum = val.x + val.y + val.z + val.w;
    partial_mean += sum;
    partial_var += val.x * val.x;
    partial_var += val.y * val.y;
    partial_var += val.z * val.z;
    partial_var += val.w * val.w;
  }
  float inv = 1.0f / N;
  float mean = warp_reduce_bcast(partial_mean) * -inv;
  float var = warp_reduce_bcast(partial_var) * inv - (mean * mean);

  float *yi = &y[row * N];
  float4 *yi4 = reinterpret_cast<float4 *>(yi);
  float rstddev = rsqrtf(var + EPSILON);

#pragma unroll
  for (int j = tx; j < values; j += 32) {
    float4 out = xi4[j];
    out.x = (out.x + mean) * rstddev;
    out.y = (out.y + mean) * rstddev;
    out.z = (out.z + mean) * rstddev;
    out.w = (out.w + mean) * rstddev;
    yi4[j] = out;
  }
}

int main() {
  constexpr std::size_t M = 2048;
  constexpr std::size_t N = 8196;
  // constexpr std::size_t M = 1;
  // constexpr std::size_t N = 2048;
  constexpr std::size_t size = M * N;

  constexpr std::size_t num_threads = 32;

  std::cout << "\n*****************" << std::endl;
  std::cout << "Launching Layer Norm: " << std::endl;
  std::cout << "  M: " << M << std::endl;
  std::cout << "  N: " << N << std::endl;
  std::cout << std::endl;

  std::vector<float> x(M * N);
  std::vector<float> y(M * N);

  std::iota(x.begin(), x.end(), 1.0f);
  // std::fill(x.begin(), x.end(), 1.0f);

  float *x_data, *y_data;
  int iterations = 1;

  cudaMalloc(&x_data, sizeof(float) * size);
  cudaMalloc(&y_data, sizeof(float) * size);

  cudaMemcpy(x_data, x.data(), sizeof(float) * size, cudaMemcpyHostToDevice);
  cudaMemset(y_data, 0, sizeof(float) * size);

  dim3 threads(num_threads);
  dim3 blocks(M);
  int shmem_bytes = num_threads * sizeof(float);

  std::vector<KernelConfig> suite = {
      KernelConfig{"Naive", layernorm_naive, blocks, threads, 0, iterations},
      KernelConfig{"1pass", layernorm_1pass, blocks, threads, shmem_bytes,
                   iterations},
      KernelConfig{"warp_reduce", layernorm_warpreduce, blocks, threads, 0,
                   iterations},
      KernelConfig{"vectorize", layernorm_vectorize, blocks, threads, 0,
                   iterations},
  };

  // Reuse host buffers you already have
  run_layernorm_suite(suite, x_data, y_data, M, N, x, y);

  std::cout << "\n\n** Exiting" << std::endl;

  cudaFree(x_data);
  cudaFree(y_data);
  return 0;
}
