#include <cuda.h>
#include <iostream>
#include <numeric>
#include <vector>

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

struct KernelConfig;
using ScanFunction = void (*)(const KernelConfig &kc, const int *, int *, int);

struct KernelConfig {
  std::string name;
  ScanFunction scan;
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

void ref_scan(const std::vector<int> &input, std::vector<int> &output,
              std::size_t size) {
  int acc = 0;
#pragma unroll
  for (std::size_t i = 0; i < size; ++i) {
    output[i] = acc;
    acc += input[i];
  }
}

void verify(const std::vector<int> &input, std::vector<int> &res_vec,
            std::size_t size, int limit = 25) {
  std::vector<int> res_ref_vec(size);
  ref_scan(input, res_ref_vec, size);
  int wrong = 0;
  for (int i = 0; i < size; ++i) {
    if (res_ref_vec[i] != res_vec[i]) {
      std::cout << "You are terrible at SCAN " << i << " C: " << res_vec[i]
                << " Ref: " << res_ref_vec[i]
                << " diff: " << std::abs(res_vec[i] - res_ref_vec[i])
                << std::endl;
      if (++wrong == limit) {
        std::cout << "Error limit (" << limit << ") reached: EXITING"
                  << std::endl;
        break;
      }
    }
  }
}

void run_scan_suite(const std::vector<KernelConfig> &ks, const int *input,
                    int *output, std::size_t size,
                    const std::vector<int> &input_host,
                    std::vector<int> &output_host) {

  for (const auto &cfg : ks) {
    printf("%-20s (%2d, %2d) (%2d, %2d)", cfg.name.c_str(), cfg.grid.x,
           cfg.grid.y, cfg.block.x, cfg.block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < cfg.iterations; i++)
      cfg.scan(cfg, input, output, size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = elapsed_ms(start, stop) / cfg.iterations;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpyAsync(output_host.data(), output, sizeof(int) * size,
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    const double flops = size;
    double bytes = 2 * size;
    const double gflops = (flops / (ms / 1000.0)) / 1e9;
    double mbytess = (bytes / (ms / 1000.0)) / 1e6;
    printf("%8.3f ms %8.2f GFLOP/s %8.2f Mbytes/s\n", ms, gflops, mbytess);

    if (size <= 2048) {
      verify(input_host, output_host, size);
    }
  }

  if (size > 2048) {
    std::cout << "  Skipping Verification (size " << size << " > 2048)"
              << std::endl;
  }
}

__global__ void scan_1st_shmem(const int *__restrict__ input, int *output,
                               int *partials, int size) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int local_x = threadIdx.x;

  const int tile_size = blockDim.x;
  extern __shared__ int shmem[];
  shmem[local_x] = (global_x < size) ? input[global_x] : 0;
  __syncthreads();
  const int v = shmem[local_x];

  for (int stride = 1; stride < tile_size; stride <<= 1) {
    if (local_x + stride < tile_size) {
      shmem[local_x + stride] += shmem[local_x];
    }
    __syncthreads();
  }

  if (global_x < size) {
    output[global_x] = shmem[local_x] - v;
  }
  if (local_x == blockDim.x - 1) {
    partials[blockIdx.x] = shmem[local_x];
  }
}

__global__ void scan_2nd_shmem(int *output, int *partials, int size) {
  if (blockIdx.x == 0)
    return;

  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int local_x = threadIdx.x;

  extern __shared__ int shmem[];
  shmem[local_x] = (local_x < blockDim.x) ? partials[local_x] : 0;
  __syncthreads();

  int acc = (global_x < size) ? output[global_x] : 0;
  for (int i = 0; i < blockIdx.x; i++) {
    acc += shmem[i];
  }

  if (global_x < size) {
    output[global_x] = acc;
  }
}

void scan_naive(const KernelConfig &cfg, const int *input, int *output,
                int size) {
  int *partial_scans;
  int partial_scans_mem = sizeof(int) * cfg.grid.x;
  cudaMalloc(&partial_scans, partial_scans_mem);

  scan_1st_shmem<<<cfg.grid, cfg.block, cfg.shared_mem_bytes>>>(
      input, output, partial_scans, size);
  if (cfg.grid.x > 1) {
    scan_2nd_shmem<<<cfg.grid, cfg.block, partial_scans_mem>>>(
        output, partial_scans, size);
  }

  cudaFree(partial_scans);
}

void scan_warp(const KernelConfig &cfg, const int *input, int *output,
               int size) {
  int *partial_scans;
  int partial_scans_mem = sizeof(int) * cfg.grid.x;
  cudaMalloc(&partial_scans, partial_scans_mem);

  scan_1st_shmem<<<cfg.grid, cfg.block, cfg.shared_mem_bytes>>>(
      input, output, partial_scans, size);
  if (cfg.grid.x > 1) {
    scan_2nd_shmem<<<cfg.grid, cfg.block, partial_scans_mem>>>(
        output, partial_scans, size);
  }

  cudaFree(partial_scans);
}

// TODO: Warp scan 1st phase
// TODO: Warp shufl reduction 2nd phase + simple addition 3rd phase
// TODO: One Sweep
// TODO: Lookback Scan
int main() {
  constexpr std::size_t num_threads = 256;
  constexpr std::size_t size = 1e6;

  // constexpr std::size_t size = 2048;

  std::cout << "\n*****************" << std::endl;
  std::cout << "Launching Scans: " << std::endl;
  std::cout << "  size: " << size << std::endl;
  std::cout << std::endl;

  std::vector<int> input(size);
  std::vector<int> output(size);

  std::fill(input.begin(), input.end(), 1);

  int *input_data, *output_data;

  cudaMalloc(&input_data, sizeof(int) * size);
  cudaMalloc(&output_data, sizeof(int) * size);

  cudaMemcpy(input_data, input.data(), sizeof(int) * size,
             cudaMemcpyHostToDevice);
  cudaMemset(output_data, 0, sizeof(int) * size);

  dim3 threads(num_threads);
  dim3 blocks(safe_div(size, threads.x));
  int shmem_size = sizeof(int) * threads.x;
  int iterations = 1;

  std::vector<KernelConfig> suite = {
      KernelConfig{"Naive", scan_naive, blocks, threads, shmem_size,
                   iterations},
  };

  // Reuse host buffers you already have
  run_scan_suite(suite, input_data, output_data, size, input, output);

  cudaFree(input_data);
  cudaFree(output_data);

  return 0;
}
