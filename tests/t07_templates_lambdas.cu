// Modern CUDA C++: template kernels (explicit and deduced), functors and device lambdas
// as kernel parameters, kernel function pointers, constexpr, namespaces.
// (These run on the CPU back end; the OpenCL translation covers the C subset.)
#include "common.h"

namespace mylib {
template <typename T, int BLOCK>
__global__ void scale(T* x, T s, int n) {
  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i < n) x[i] *= s;
}

template <typename T, typename F>
__global__ void transform(const T* in, T* out, int n, F f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = f(in[i]);
}

struct Square {
  __device__ float operator()(float v) const { return v * v; }
};
}  // namespace mylib

template <typename T>
__global__ void iota(T* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = static_cast<T>(i);
}

__global__ void addOne(int* p) { p[threadIdx.x] += 1; }

constexpr int kN = 1000;

int main() {
  double* dd;
  float *fa, *fb;
  CHECK(cudaMalloc(&dd, kN * sizeof(double)));
  CHECK(cudaMalloc(&fa, kN * sizeof(float)));
  CHECK(cudaMalloc(&fb, kN * sizeof(float)));
  iota<<<(kN + 127) / 128, 128>>>(dd, kN);   // deduced template argument
  mylib::scale<double, 64><<<(kN + 63) / 64, 64>>>(dd, 0.5, kN);
  double hd[kN];
  CHECK(cudaMemcpy(hd, dd, sizeof hd, cudaMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) EXPECT(hd[i] == 0.5 * i, "scale<double>[%d] = %f", i, hd[i]);

  iota<float><<<(kN + 255) / 256, 256>>>(fa, kN);
  mylib::transform<<<(kN + 255) / 256, 256>>>(fa, fb, kN, mylib::Square());
  float hf[kN];
  CHECK(cudaMemcpy(hf, fb, sizeof hf, cudaMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) EXPECT(hf[i] == (float)i * (float)i, "functor[%d]", i);

  float k = 3.0f;
  auto lam = [=] __device__(float v) { return v * k + 1.0f; };
  mylib::transform<<<(kN + 255) / 256, 256>>>(fa, fb, kN, lam);
  CHECK(cudaMemcpy(hf, fb, sizeof hf, cudaMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) EXPECT(hf[i] == (float)i * 3.0f + 1.0f, "lambda[%d]", i);

  void (*kp)(int*) = addOne;
  int* di;
  CHECK(cudaMalloc(&di, 32 * sizeof(int)));
  CHECK(cudaMemset(di, 0, 32 * sizeof(int)));
  kp<<<1, 32>>>(di);
  kp<<<1, 32>>>(di);
  int hi[32];
  CHECK(cudaMemcpy(hi, di, sizeof hi, cudaMemcpyDeviceToHost));
  for (int i = 0; i < 32; ++i) EXPECT(hi[i] == 2, "fnptr[%d] = %d", i, hi[i]);
  printf("PASS t07_templates_lambdas\n");
  return 0;
}
