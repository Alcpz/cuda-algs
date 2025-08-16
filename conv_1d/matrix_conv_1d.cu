#include <algorithm>
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

using ConvKernel = void (*)(const float *, const float *, float *, int, int,
                            int, int);

struct KernelConfig {
  std::string name;
  ConvKernel kernel;
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

void conv1d_ref(const std::vector<float> &x, // [B*T*C]
                const std::vector<float> &w, // [C*K]
                std::vector<float> &y,       // [B*T*C]
                std::size_t B, std::size_t C, std::size_t T, std::size_t K) {
  for (size_t b = 0; b < B; ++b) {
    // printf("\n\nB = %lu", b);
    for (size_t c = 0; c < C; ++c) {
      // printf("\n");
      for (size_t t = 0; t < T; ++t) {
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          int tt = t - k;
          int kk = K - 1 - k;
          if (tt >= 0) {
            acc += x[((b * C) + c) * T + tt] * w[c * K + kk];
          }
        }
        y[((b * C) + c) * T + t] = acc;
        // printf("%.2f ", y[((b * C) + c) * T + t]);
      }
    }
  }
  // printf("\n");
}

void verify(std::vector<float> x, std::vector<float> w,
            std::vector<float> y_vec, std::size_t B, std::size_t C,
            std::size_t T, std::size_t K, int limit = 25) {
  std::vector<float> y_ref_vec(B * C * T);
  conv1d_ref(x, w, y_ref_vec, B, C, T, K);
  int wrong = 0;
  for (int i = 0; i < B * C * T; ++i) {
    if (!isAlmostEqual(y_ref_vec[i], y_vec[i])) {
      std::cout << "You are terrible at CONV " << i << " C: " << y_vec[i]
                << " Ref: " << y_ref_vec[i]
                << " diff: " << std::abs(y_vec[i] - y_ref_vec[i]) << std::endl;
      if (++wrong == limit) {
        std::cout << "Error limit (" << limit << ") reached: EXITING"
                  << std::endl;
        break;
      }
    }
  }
}

void run_conv_suite(const std::vector<KernelConfig> &ks, const float *x,
                    const float *w, float *y, std::size_t B, std::size_t C,
                    std::size_t T, std::size_t K,
                    const std::vector<float> &x_host,
                    const std::vector<float> &w_host,
                    std::vector<float> &y_host) {

  for (const auto &cfg : ks) {
    printf("%-20s (%2d, %2d, %2d) (%2d, %2d, %2d)", cfg.name.c_str(),
           cfg.grid.x, cfg.grid.y, cfg.grid.z, cfg.block.x, cfg.block.y,
           cfg.block.z);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < cfg.iterations; i++)
      cfg.kernel<<<cfg.grid, cfg.block, cfg.shared_mem_bytes>>>(x, w, y, B, C,
                                                                T, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = elapsed_ms(start, stop) / cfg.iterations;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpyAsync(y_host.data(), y,
                               sizeof(float) * std::size_t(B * C * T),
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    // K - 1 comes from K = 3, first two values need 1 and 2 W
    const double flops = (2 * B * C * T * K) - 2 - 1;
    const double gflops = (flops / (ms / 1000.0)) / 1e9;
    const double bytes = 2 * B * C * T + C * K;
    const double gbytess = (bytes / (ms / 1000.0)) / 1e9;
    printf("%8.3f ms  %8.2f GFLOP/s  %8.2f GBytes/s\n", ms, gflops, gbytess);

    if (C * B * T <= 2048) {
      verify(x_host, w_host, y_host, B, C, T, K);
    }
  }

  if (C * B * T > 2048) {
    std::cout << "  Skipping Verification (size " << B * C * T << " > 2048)"
              << std::endl;
  }
}

__global__ void conv_1d(const float *__restrict__ x,
                        const float *__restrict__ w, float *y, int B, int C,
                        int T, int K) {
  int tx = threadIdx.x;
  int row = (blockIdx.x * gridDim.y) + blockIdx.y;
  int c = blockIdx.y;

  if (tx >= T)
    return;

  const float *xi = &x[row * T];
  const float *wj = &w[c * K];

  for (int idx = tx; idx < T; idx += blockDim.x) {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < K; k++) {
      int tt = idx - k;
      int kk = K - 1 - k;
      if (idx - k >= 0) {
        acc += xi[tt] * wj[kk];
      }
    }
    y[row * T + idx] = acc;
  }
}

__global__ void conv_1d_shmem_all(const float *__restrict__ x,
                                  const float *__restrict__ w, float *y, int B,
                                  int C, int T, int K) {
  int tx = threadIdx.x;
  int row = (blockIdx.x * gridDim.y) + blockIdx.y;
  // Blocks(B, C)
  int c = blockIdx.y;

  const float *xi = &x[row * T];
  const float *wj = &w[c * K];

  extern __shared__ float shmem[];

  for (int idx = tx; idx < T; idx += blockDim.x) {
    shmem[idx] = xi[idx];
  }
  __syncthreads();

  if (tx >= T)
    return;

  for (int idx = tx; idx < T; idx += blockDim.x) {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < K; k++) {
      int tt = idx - k;
      int kk = K - 1 - k;
      if (idx - k >= 0) {
        acc += shmem[tt] * wj[kk];
      }
    }
    y[row * T + idx] = acc;
  }
}

__constant__ float c_kern[3 * 2048];

__global__ void conv_1d_cmem(const float *__restrict__ x,
                             const float *__restrict__ w, float *y, int B,
                             int C, int T, int K) {
  int tx = threadIdx.x;
  int row = (blockIdx.x * gridDim.y) + blockIdx.y;
  // Blocks(B, C)
  int c = blockIdx.y;

  const float *xi = &x[row * T];
  const float *wj = &c_kern[c * K];

  extern __shared__ float shmem[];

  for (int idx = tx; idx < T; idx += blockDim.x) {
    shmem[idx] = xi[idx];
  }
  __syncthreads();

  if (tx >= T)
    return;

  for (int idx = tx; idx < T; idx += blockDim.x) {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < K; k++) {
      int tt = idx - k;
      int kk = K - 1 - k;
      if (idx - k >= 0) {
        acc += shmem[tt] * wj[kk];
      }
    }
    y[row * T + idx] = acc;
  }
}

__global__ void conv_1d_vectorized(const float *__restrict__ x,
                                   const float *__restrict__ w, float *y, int B,
                                   int C, int T, int K) {
  int tx = threadIdx.x;
  int row = (blockIdx.x * gridDim.y) + blockIdx.y;
  // Blocks(B, C)
  int c = blockIdx.y;

  const float *xi = &x[row * T];
  const float4 *xi4 = reinterpret_cast<const float4 *>(xi);
  const float *wj = &c_kern[c * K];

  extern __shared__ float shmem[];
  float4 *shmem4 = reinterpret_cast<float4 *>(shmem);
  int vec_count = T / 4;

  for (int idx = tx; idx < vec_count; idx += blockDim.x) {
    shmem4[idx] = xi4[idx];
  }
  __syncthreads();

  if (tx >= T)
    return;

  for (int idx = tx; idx < T; idx += blockDim.x) {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < K; k++) {
      int tt = idx - k;
      int kk = K - 1 - k;
      if (tt >= 0) {
        acc += shmem[tt] * wj[kk];
      }
    }
    y[row * T + idx] = acc;
  }
}

__global__ void conv_1d_reg_tile(const float *__restrict__ x,
                                 const float *__restrict__ w, float *y, int B,
                                 int C, int T, int K) {
  int tx = threadIdx.x;
  // Blocks(B, C)
  int b = blockIdx.x;
  int c = blockIdx.y;
  if (b >= B || c >= C) {
    return;
  }

  int row = (b * C) + c;

  const float *xi = &x[row * T];
  const float4 *xi4 = reinterpret_cast<const float4 *>(xi);
  const float *wj = &c_kern[c * K];
  float4 *y4 = reinterpret_cast<float4 *>(&y[row * T]);

  extern __shared__ float shmem[];
  float4 *shmem4 = reinterpret_cast<float4 *>(shmem);
  int vec_count = T / 4;

  for (int idx = tx; idx < vec_count; idx += blockDim.x) {
    shmem4[idx] = xi4[idx];
  }
  __syncthreads();

  if (tx >= vec_count)
    return;

  float regs[4];
  for (int idx = tx; idx < vec_count; idx += blockDim.x) {
    int offset = idx * 4;
    float4 out;
    out.x = 0.0f;
    out.y = 0.0f;
    out.z = 0.0f;
    out.w = 0.0f;
    regs[0] = shmem[offset];
    regs[1] = shmem[offset + 1];
    regs[2] = shmem[offset + 2];
    regs[3] = shmem[offset + 3];
#pragma unroll
    for (int k = 0; k < K; k++) {
      int kk = K - 1 - k;
      const float wjk = wj[kk];
      out.x += regs[0] * wjk;
      out.y += regs[1] * wjk;
      out.z += regs[2] * wjk;
      out.w += regs[3] * wjk;

      regs[3] = regs[2];
      regs[2] = regs[1];
      regs[1] = regs[0];
      regs[0] = offset - (k + 1) >= 0 ? shmem[offset - (k + 1)] : 0.0f;
    }
    y4[idx] = out;
  }
}

__global__ void conv_1d_k3(const float *__restrict__ x,
                           const float *__restrict__ w, float *y, int B, int C,
                           int T, int K) {
  constexpr int KC = 3;

  // Blocks(B, C)
  int b = blockIdx.x;
  int c = blockIdx.y;
  if (b >= B || c >= C) {
    return;
  }

  int tx = threadIdx.x;
  int row = (b * C) + c;

  const float *xi = &x[row * T];
  const float4 *xi4 = reinterpret_cast<const float4 *>(xi);
  const float *wj = &c_kern[c * KC];
  float4 *y4 = reinterpret_cast<float4 *>(&y[row * T]);

  int vec_count = T / 4;
  if (tx >= vec_count)
    return;

  float kern0 = wj[0];
  float kern1 = wj[1];
  float kern2 = wj[2];

#pragma unroll
  for (int idx = tx; idx < vec_count; idx += blockDim.x) {
    int offset = idx * 4;
    float4 out, v;
    float xm1, xm2;

    v = xi4[idx];
    if (idx > 0) {
      float4 v_prev = xi4[idx - 1];
      xm1 = v_prev.w;
      xm2 = v_prev.z;
    } else {
      xm1 = offset - 1 >= 0 ? xi[offset - 1] : 0.0f;
      xm2 = offset - 2 >= 0 ? xi[offset - 2] : 0.0f;
    }

    out.x = xm2 * kern0 + xm1 * kern1 + v.x * kern2;
    out.y = xm1 * kern0 + v.x * kern1 + v.y * kern2;
    out.z = v.x * kern0 + v.y * kern1 + v.z * kern2;
    out.w = v.y * kern0 + v.z * kern1 + v.w * kern2;

    y4[idx] = out;
  }
}

int main() {

  constexpr std::size_t num_threads = 256;

  constexpr std::size_t K = 3;
  constexpr std::size_t C = 1024;
  constexpr std::size_t B = 32;
  constexpr std::size_t T = 2048;

  // constexpr std::size_t C = 1;
  // constexpr std::size_t B = 1;
  // constexpr std::size_t T = 256;
  // constexpr std::size_t K = 3;

  constexpr std::size_t conv_size = B * C * T;
  constexpr std::size_t kern_size = C * K;
  constexpr std::size_t res_size = B * C * T;

  constexpr std::size_t conv_size_bytes = conv_size * sizeof(float);
  constexpr std::size_t kern_size_bytes = kern_size * sizeof(float);
  constexpr std::size_t res_size_bytes = res_size * sizeof(float);

  constexpr std::size_t conv1d_size_bytes = T * sizeof(float);

  // constexpr std::size_t conv_size = 1024;
  // constexpr std::size_t kern_size = 12;

  std::cout << "\n*****************" << std::endl;
  std::cout << "Launching Convs: " << std::endl;
  std::cout << "  conv_size: " << B << " " << C << " " << T << std::endl;
  std::cout << "  kern_size: " << C << " 1 " << K << std::endl;
  std::cout << "  res_size: " << B << " " << C << " " << T << std::endl;
  std::cout << std::endl;

  std::vector<float> values(conv_size);
  std::vector<float> kern(kern_size);
  std::vector<float> res(res_size);

  for (std::size_t b = 0; b < B; b++) {
    for (std::size_t c = 0; c < C; c++) {
      auto row = values.begin() + (b * C + c) * T;
      std::iota(row, row + T, 0.0f);
    }
  }
  for (std::size_t c = 0; c < C; c++) {
    auto row = kern.begin() + c * K;
    std::iota(row, row + K, 0.0f);
  }

  float *kern_data, *values_data, *res_data;

  cudaMalloc(&values_data, conv_size_bytes);
  cudaMalloc(&kern_data, kern_size_bytes);
  cudaMalloc(&res_data, res_size_bytes);

  cudaMemcpy(values_data, values.data(), conv_size_bytes,
             cudaMemcpyHostToDevice);
  cudaMemcpy(kern_data, kern.data(), kern_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(c_kern, kern.data(), kern_size_bytes);
  cudaMemset(res_data, 0, res_size_bytes);

  dim3 blocks(B, C);
  dim3 threads(num_threads);
  int iterations = 10;
  int shmem_size = conv1d_size_bytes;

  std::vector<KernelConfig> suite = {
      KernelConfig{"Naive", conv_1d, blocks, threads, 0, iterations},
      KernelConfig{"Shmem", conv_1d_shmem_all, blocks, threads, shmem_size,
                   iterations},
      KernelConfig{"Cmem", conv_1d_cmem, blocks, threads, shmem_size,
                   iterations},
      KernelConfig{"Vectorized", conv_1d_vectorized, blocks, threads,
                   shmem_size, iterations},
      KernelConfig{"Reg_tile", conv_1d_reg_tile, blocks, threads, shmem_size,
                   iterations},
      KernelConfig{"Vec_k=3", conv_1d_k3, blocks, threads, 0, iterations},
  };

  // Reuse host buffers you already have
  run_conv_suite(suite, values_data, kern_data, res_data, B, C, T, K, values,
                 kern, res);

  cudaFree(values_data);
  cudaFree(kern_data);
  cudaFree(res_data);

  return 0;
}
