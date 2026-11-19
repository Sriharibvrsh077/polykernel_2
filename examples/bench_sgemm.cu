// Single-precision matrix multiply benchmark (C = A * B, square N x N, row major).
// Plain CUDA: builds unchanged with nvcc and with PolyKernel's pkcc, so the same source can be
// timed natively on an NVIDIA GPU and through PolyKernel (CPU or OpenCL) on any hardware.
//
//   bench_sgemm [N=1024] [reps=5] [kernels=naive,tiled,reg]
//
// Each kernel is warmed up for 300 ms, then timed as the best of 3 rounds of `reps` multiplies.
// Prints one "RESULT" line per kernel (milliseconds per multiply, GFLOP/s, max relative error
// against a double-precision reference on 512 sampled entries) and PASS/FAIL at the end.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

#define CHECK(x)                                                                         \
  do {                                                                                   \
    cudaError_t e_ = (x);                                                                \
    if (e_ != cudaSuccess) {                                                             \
      printf("FAIL: %s at line %d: %s\n", #x, __LINE__, cudaGetErrorString(e_));         \
      exit(1);                                                                           \
    }                                                                                    \
  } while (0)

// One thread per output element, no reuse.
__global__ void sgemm_naive(const float* A, const float* B, float* C, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < N && col < N) {
    float acc = 0.f;
    for (int k = 0; k < N; ++k) acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// 16 x 16 tiles staged through shared memory (the classic textbook kernel).
#define TS 16
__global__ void sgemm_tiled(const float* A, const float* B, float* C, int N) {
  __shared__ float As[TS][TS];
  __shared__ float Bs[TS][TS];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * TS + ty, col = blockIdx.x * TS + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < N; k0 += TS) {
    As[ty][tx] = (row < N && k0 + tx < N) ? A[row * N + k0 + tx] : 0.f;
    Bs[ty][tx] = (k0 + ty < N && col < N) ? B[(k0 + ty) * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TS; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < N && col < N) C[row * N + col] = acc;
}

// 64 x 64 block tile, 256 threads, each thread accumulates a 4 x 4 register tile
// (rows ty + 16 i, columns tx + 16 j, so shared-memory reads and global stores stay contiguous).
// Requires N to be a multiple of 64.
#define BM 64
#define BN 64
#define BK 16
__global__ void sgemm_reg(const float* A, const float* B, float* C, int N) {
  __shared__ float As[BK][BM];   // A tile stored transposed
  __shared__ float Bs[BK][BN];
  int t = threadIdx.x;
  int tx = t % 16, ty = t / 16;
  int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
  for (int k0 = 0; k0 < N; k0 += BK) {
#pragma unroll
    for (int l = t; l < BM * BK; l += 256) {
      int ar = l / BK, ac = l % BK;
      As[ac][ar] = A[(row0 + ar) * N + k0 + ac];
      int br = l / BN, bc = l % BN;
      Bs[br][bc] = B[(k0 + br) * N + col0 + bc];
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = As[k][ty + 16 * i];
#pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = Bs[k][tx + 16 * j];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) C[(row0 + ty + 16 * i) * N + col0 + tx + 16 * j] = acc[i][j];
}

static unsigned long long rng = 0x9E3779B97F4A7C15ull;
static float frand() {   // xorshift64*, uniform in [-1, 1)
  rng ^= rng >> 12;
  rng ^= rng << 25;
  rng ^= rng >> 27;
  return (float)((rng * 2685821657736338717ull) >> 40) / (float)(1 << 23) - 1.f;
}

int main(int argc, char** argv) {
  int N = argc > 1 ? atoi(argv[1]) : 1024;
  int reps = argc > 2 ? atoi(argv[2]) : 5;
  const char* which = argc > 3 ? argv[3] : "naive,tiled,reg";
  if (N <= 0 || reps <= 0) {
    printf("usage: bench_sgemm [N] [reps] [kernels]\n");
    return 2;
  }
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("device: %s\nN=%d reps=%d\n", prop.name, N, reps);

  size_t n2 = (size_t)N * N;
  std::vector<float> hA(n2), hB(n2), hC(n2);
  for (size_t i = 0; i < n2; ++i) hA[i] = frand();
  for (size_t i = 0; i < n2; ++i) hB[i] = frand();
  float *dA, *dB, *dC;
  CHECK(cudaMalloc(&dA, n2 * sizeof(float)));
  CHECK(cudaMalloc(&dB, n2 * sizeof(float)));
  CHECK(cudaMalloc(&dC, n2 * sizeof(float)));
  CHECK(cudaMemcpy(dA, hA.data(), n2 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, hB.data(), n2 * sizeof(float), cudaMemcpyHostToDevice));

  // reference values for 512 sampled entries, in double precision
  const int S = 512;
  std::vector<int> si(S), sj(S);
  std::vector<double> ref(S);
  for (int s = 0; s < S; ++s) {
    si[s] = (int)((frand() + 1.f) * 0.5f * N) % N;
    sj[s] = (int)((frand() + 1.f) * 0.5f * N) % N;
    double acc = 0;
    for (int k = 0; k < N; ++k) acc += (double)hA[(size_t)si[s] * N + k] * hB[(size_t)k * N + sj[s]];
    ref[s] = acc;
  }

  cudaEvent_t e0, e1;
  CHECK(cudaEventCreate(&e0));
  CHECK(cudaEventCreate(&e1));
  bool all_ok = true;
  const char* names[3] = {"naive", "tiled", "reg"};
  for (int kidx = 0; kidx < 3; ++kidx) {
    const char* kn = names[kidx];
    if (!strstr(which, kn)) continue;
    if (kidx == 2 && N % 64) {
      printf("RESULT kernel=%s skipped (N must be a multiple of 64)\n", kn);
      continue;
    }
    auto launch = [&]() {
      if (kidx == 0) {
        dim3 b(16, 16), g((N + 15) / 16, (N + 15) / 16);
        sgemm_naive<<<g, b>>>(dA, dB, dC, N);
      } else if (kidx == 1) {
        dim3 b(TS, TS), g((N + TS - 1) / TS, (N + TS - 1) / TS);
        sgemm_tiled<<<g, b>>>(dA, dB, dC, N);
      } else {
        dim3 b(256), g(N / BN, N / BM);
        sgemm_reg<<<g, b>>>(dA, dB, dC, N);
      }
    };
    CHECK(cudaMemset(dC, 0, n2 * sizeof(float)));
    // Warm up for at least 300 ms: GPUs idle at low clocks and need a moment to reach full speed,
    // so whichever program runs first would otherwise look slower.
    launch();
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    {
      auto w0 = std::chrono::steady_clock::now();
      while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < 0.3) {
        launch();
        CHECK(cudaDeviceSynchronize());
      }
    }
    float ms = 1e30f;
    for (int round = 0; round < 3; ++round) {   // best of 3 rounds of `reps` multiplies
      CHECK(cudaEventRecord(e0));
      for (int r = 0; r < reps; ++r) launch();
      CHECK(cudaEventRecord(e1));
      CHECK(cudaEventSynchronize(e1));
      CHECK(cudaGetLastError());
      float t = 0.f;
      CHECK(cudaEventElapsedTime(&t, e0, e1));
      if (t / reps < ms) ms = t / reps;
    }
    CHECK(cudaMemcpy(hC.data(), dC, n2 * sizeof(float), cudaMemcpyDeviceToHost));
    double maxrel = 0;
    for (int s = 0; s < S; ++s) {
      double got = hC[(size_t)si[s] * N + sj[s]];
      double rel = fabs(got - ref[s]) / (fabs(ref[s]) + 1.0);
      if (rel > maxrel) maxrel = rel;
    }
    bool ok = maxrel < 1e-3;
    all_ok = all_ok && ok;
    double gflops = 2.0 * N * (double)N * N / (ms * 1e6);
    printf("RESULT kernel=%s N=%d ms=%.3f gflops=%.2f maxrelerr=%.2e %s\n", kn, N, ms, gflops, maxrel,
           ok ? "OK" : "WRONG");
  }
  CHECK(cudaEventDestroy(e0));
  CHECK(cudaEventDestroy(e1));
  CHECK(cudaFree(dA));
  CHECK(cudaFree(dB));
  CHECK(cudaFree(dC));
  printf("%s bench_sgemm\n", all_ok ? "PASS" : "FAIL");
  return all_ok ? 0 : 1;
}
