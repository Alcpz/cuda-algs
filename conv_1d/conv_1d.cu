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

using ConvKernel = void (*)(const float *, const float *, float *, int, int);

struct KernelConfig {
  std::string name;
  ConvKernel kernel;
  dim3 grid;
  dim3 block;
  int shared_mem_bytes;
};

static inline float elapsed_ms(cudaEvent_t s, cudaEvent_t e) {
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
  return ms;
}

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

inline constexpr int res_size(int conv_size, int kern_size) {
  return conv_size - kern_size + 1;
}

void ref_convolution(const std::vector<float> &values,
                     const std::vector<float> &kern, std::vector<float> &res,
                     std::size_t size, std::size_t k_size) {
  const std::size_t out_len = res_size(size, k_size);

  for (std::size_t i = 0; i < out_len; ++i) {
    float acc = 0.0f;
    // flip kernel: kern[k_size - 1 - k]
    for (std::size_t k = 0; k < k_size; ++k) {
      acc += values[i + k] * kern[k];
    }
    res[i] = acc;
  }
}

template <typename T>
void verify(std::vector<T> values, std::vector<T> kern, std::vector<T> res_vec,
            std::size_t size, std::size_t k_size, int limit = 25) {
  std::vector<float> res_ref_vec(size - k_size + 1);
  ref_convolution(values, kern, res_ref_vec, size, k_size);
  int wrong = 0;
  for (int i = 0; i < size - k_size + 1; ++i) {
    if (!isAlmostEqual(res_ref_vec[i], res_vec[i])) {
      std::cout << "You are terrible at CONV " << i << " C: " << res_vec[i]
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

void run_conv_suite(const std::vector<KernelConfig> &ks, const float *values,
                    const float *kern, float *res, std::size_t size,
                    std::size_t k_size, const std::vector<float> &values_host,
                    const std::vector<float> &kern_host,
                    std::vector<float> &res_host) {

  for (const auto &cfg : ks) {
    printf("%-20s (%2d, %2d) (%2d, %2d)", cfg.name.c_str(), cfg.grid.x,
           cfg.grid.y, cfg.block.x, cfg.block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++)
      cfg.kernel<<<cfg.grid, cfg.block, cfg.shared_mem_bytes>>>(
          values, kern, res, size, k_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = elapsed_ms(start, stop) / 100;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpyAsync(res_host.data(), res,
                               sizeof(float) * std::size_t(size - k_size + 1),
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Report GFLOP/s (2*M*N*K flops)
    const double flops = 2 * k_size * (size - k_size + 1);
    const double gflops = (flops / (ms / 1000.0)) / 1e9;
    printf("%8.3f ms   %8.2f GFLOP/s\n", ms, gflops);

    if (size <= 2048) {
      verify(values_host, kern_host, res_host, size, k_size);
    }
  }

  if (size > 2048) {
    std::cout << "  Skipping Verification (size " << size << " > 2048)"
              << std::endl;
  }
}

__global__ void conv_1d(const float *__restrict__ values,
                        const float *__restrict__ kern, float *res, int size,
                        int k_size) {
  int out_size = size - k_size + 1;

  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  float acc = 0.0f;
  for (int j = 0; j < k_size; ++j) {
    if (tx + j < size)
      acc += values[tx + j] * kern[j];
  }
  if (tx < out_size) {
    res[tx] = acc;
  }
}

__global__ void conv_1d_shmem(const float *__restrict__ values,
                              const float *__restrict__ kern, float *res,
                              int size, int k_size) {
  int out_size = size - k_size + 1;

  extern __shared__ float shmem[];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x;

  shmem[tx] = idx < size ? values[idx] : 0.0f;
  if (tx < k_size - 1) {
    shmem[blockDim.x + tx] =
        (idx + blockDim.x) < size ? values[blockDim.x + idx] : 0.0f;
  }
  __syncthreads();

  float acc = 0.0f;
  if (idx < out_size) {
    for (int j = 0; j < k_size; ++j) {
      acc += shmem[tx + j] * kern[j];
    }
    res[idx] = acc;
  }
}

constexpr int MAX_K = 2048;
__constant__ float c_kern[MAX_K];

__global__ void conv_1d_cmem(const float *__restrict__ values,
                             const float *__restrict__ kern, float *res,
                             int size, int k_size) {
  int out_size = size - k_size + 1;

  extern __shared__ float shmem[];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x;

  shmem[tx] = idx < size ? values[idx] : 0.0f;
  if (tx < k_size - 1) {
    shmem[blockDim.x + tx] =
        (idx + blockDim.x) < size ? values[blockDim.x + idx] : 0.0f;
  }
  __syncthreads();

  float acc = 0.0f;
  if (idx < out_size) {
    for (int j = 0; j < k_size; ++j) {
      acc += shmem[tx + j] * c_kern[j];
    }
    res[idx] = acc;
  }
}

constexpr int RACC = 2;
__global__ void conv_1d_regtile(const float *__restrict__ values,
                                const float *__restrict__ kern, float *res,
                                int size, int k_size) {
  int out_size = size - k_size + 1;
  extern __shared__ float shmem[];

  int shmem_size = RACC * blockDim.x + k_size - 1;

  int tx = threadIdx.x * RACC;
  int gx = (RACC * blockIdx.x * blockDim.x);

  for (int lidx = threadIdx.x; lidx < shmem_size; lidx += blockDim.x) {
    const int offset = gx + lidx;
    shmem[lidx] = offset < size ? values[offset] : 0.0f;
  }
  __syncthreads();

  if (gx + tx >= out_size)
    return;

  float acc[RACC] = {0.0f};
  float vals[RACC];
  // Needs to avoid bank conflicts
  for (int r = 0; r < RACC; ++r) {
    vals[r] = shmem[tx + r];
  }

  for (int j = 0; j < k_size; ++j) {
    const float c_k_val = c_kern[j];
    for (int r = 0; r < RACC; ++r) {
      acc[r] = fmaf(vals[r], c_k_val, acc[r]);
    }
    if (j + 1 < k_size) {
      for (int r = 0; r < RACC - 1; ++r) {
        vals[r] = vals[r + 1];
      }
      vals[RACC - 1] = shmem[tx + j + RACC];
    }
  }

  res[gx + tx] = acc[0];
  if (gx + tx + 1 < out_size)
    res[gx + tx + 1] = acc[1];
}

__global__ void conv_1d_vload(const float *__restrict__ values,
                              const float *__restrict__ kern, float *res,
                              int size, int k_size) {
  int out_size = size - k_size + 1;

  int shared_mem_size = blockDim.x + k_size - 1;
  extern __shared__ float shmem[];

  int block_offset = blockIdx.x * blockDim.x;
  int global_i = blockIdx.x * blockDim.x + threadIdx.x;

  for (int load_idx = threadIdx.x * 4;
       load_idx < shared_mem_size && load_idx + block_offset < size;
       load_idx += blockDim.x * 4) {
    int global_offset = block_offset + load_idx;

    const float4 data =
        *reinterpret_cast<const float4 *>(&values[global_offset]);
    *reinterpret_cast<float4 *>(&shmem[load_idx]) = data;
  }
  __syncthreads();

  if (global_i < out_size) {
    float acc = 0.0f;
    for (int j = 0; j < k_size; ++j) {
      acc += shmem[threadIdx.x + j] * c_kern[j];
    }
    res[global_i] = acc;
  }
}

__constant__ float c_u[4];

__global__ void conv_1d_winograd(const float *__restrict__ x,
                                 const float *__restrict__,
                                 float *__restrict__ y, int n, int) {
  // out_len = n - 3 + 1 (valid)
  const int out_len = n - 2; // since k=3
  const int tx = threadIdx.x;
  const int tid = blockIdx.x * blockDim.x + tx;

  // This thread computes two outputs: base and base+1
  const int base_out = tid * 2;
  if (base_out >= out_len)
    return;

  // Each block loads a contiguous span of inputs into smem:
  // span = 2*blockDim.x + 3 (four inputs per thread, overlap handled)
  extern __shared__ float smem[];
  const int block_in_start = blockIdx.x * (blockDim.x * 2);
  const int span = 2 * blockDim.x + 3;

  // Coalesced loads: two per thread
  int p0 = 2 * tx;
  int g0 = block_in_start + p0;
  if (p0 < span)
    smem[p0] = (g0 < n) ? x[g0] : 0.0f;

  int p1 = p0 + 1;
  int g1 = block_in_start + p1;
  if (p1 < span)
    smem[p1] = (g1 < n) ? x[g1] : 0.0f;

  // Tail (3 extra) by first 3 threads
  if (tx < 3) {
    int p = 2 * blockDim.x + tx;
    int g = block_in_start + p;
    smem[p] = (g < n) ? x[g] : 0.0f;
  }
  __syncthreads();

  // Local 4-sample tile for this thread
  float d0 = smem[2 * tx + 0];
  float d1 = smem[2 * tx + 1];
  float d2 = smem[2 * tx + 2];
  float d3 = smem[2 * tx + 3];

  // Input transform: v = B^T d
  // B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
  float v0 = d0 - d2;
  float v1 = d1 + d2;
  float v2 = d2 - d1;
  float v3 = d1 - d3;

  // Elementwise multiply with transformed kernel u = G g
  float m0 = c_u[0] * v0;
  float m1 = c_u[1] * v1;
  float m2 = c_u[2] * v2;
  float m3 = c_u[3] * v3;

  // Output transform: y = A^T m
  // A^T = [[1,1,1,0],[0,1,-1,-1]]
  float y0 = m0 + m1 + m2;
  float y1 = m1 - m2 - m3;

  y[base_out] = y0;
  if (base_out + 1 < out_len)
    y[base_out + 1] = y1; // tail-safe for odd out_len
}

int main() {

  constexpr std::size_t num_threads = 256;
  constexpr std::size_t conv_size = 2048;
  constexpr std::size_t kern_size = 3;

  // constexpr std::size_t conv_size = 1024;
  // constexpr std::size_t kern_size = 12;

  std::cout << "\n*****************" << std::endl;
  std::cout << "Launching Convs: " << std::endl;
  std::cout << "  conv_size: " << conv_size << std::endl;
  std::cout << "  kern_size: " << kern_size << std::endl;
  std::cout << std::endl;

  std::vector<float> values(conv_size);
  std::vector<float> kern(kern_size);
  std::vector<float> res(conv_size - kern_size + 1);

  std::iota(values.begin(), values.end(), 0.0f);
  std::iota(kern.begin(), kern.end(), 0.0f);

  float *kern_data, *values_data, *res_data;

  cudaMalloc(&values_data, sizeof(float) * conv_size);
  cudaMalloc(&kern_data, sizeof(float) * kern_size);
  cudaMalloc(&res_data, sizeof(float) * res_size(conv_size, kern_size));

  cudaMemcpy(kern_data, kern.data(), sizeof(float) * kern_size,
             cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(c_kern, kern.data(), sizeof(float) * kern_size);
  cudaMemcpy(values_data, values.data(), sizeof(float) * conv_size,
             cudaMemcpyHostToDevice);
  cudaMemset(res_data, 0, sizeof(float) * res_size(conv_size, kern_size));

  dim3 threads(num_threads);
  dim3 blocks(safe_div(conv_size, threads.x));
  int shmem_size = sizeof(float) * (num_threads + kern_size - 1);

  dim3 threads_reg(num_threads);
  dim3 blocks_reg(safe_div(conv_size, threads.x * RACC));
  int shmem_regs = sizeof(float) * (num_threads * RACC + kern_size - 1);

  dim3 threads_vload(num_threads);
  dim3 blocks_vload(safe_div(conv_size, threads.x));
  int shmem_vload = sizeof(float) * (num_threads + kern_size - 1);

  float u[4];
  // G = [[1,0,0],[1/2,1/2,1/2],[1/2,-1/2,1/2],[0,0,1]]
  u[0] = kern[0];
  u[1] = 0.5f * (kern[0] + kern[1] + kern[2]);
  u[2] = 0.5f * (kern[0] - kern[1] + kern[2]);
  u[3] = kern[2];
  cudaMemcpyToSymbol(c_u, u, 4 * sizeof(float));

  const int tiles = safe_div(res_size(conv_size, kern_size), kern_size - 1);
  dim3 threads_wino(num_threads);
  dim3 blocks_wino = (tiles + num_threads - 1) / num_threads;
  const size_t shmem_wino = (2 * num_threads + 3) * sizeof(float);

  std::vector<KernelConfig> suite = {
      KernelConfig{"Naive", conv_1d, blocks, threads, 0},
      KernelConfig{"Shmem", conv_1d_shmem, blocks, threads, shmem_size},
      KernelConfig{"Shmem + Cmem", conv_1d_cmem, blocks, threads, shmem_size},
      KernelConfig{"Regs", conv_1d_regtile, blocks_reg, threads_reg,
                   shmem_regs},
      KernelConfig{"VecLoads", conv_1d_vload, blocks_vload, threads_vload,
                   shmem_vload},
      KernelConfig{"Winograd", conv_1d_winograd, blocks_wino, threads_wino,
                   shmem_wino},
  };

  // Reuse host buffers you already have
  run_conv_suite(suite, values_data, kern_data, res_data, conv_size, kern_size,
                 values, kern, res);

  cudaFree(values_data);
  cudaFree(kern_data);
  cudaFree(res_data);

  return 0;
}
