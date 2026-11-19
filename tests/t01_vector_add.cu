// Basic launch, memory management, pointer arithmetic on device pointers.
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK(x)                                                                          \
  do {                                                                                    \
    cudaError_t e_ = (x);                                                                 \
    if (e_ != cudaSuccess) {                                                              \
      printf("FAIL: %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_));  \
      return 1;                                                                           \
    }                                                                                     \
  } while (0)

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void saxpy(size_t n, float alpha, const float* __restrict__ x, float* __restrict__ y) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)blockDim.x * gridDim.x)
    y[i] = alpha * x[i] + y[i];
}

int main() {
  const int n = 100003;
  const size_t bytes = n * sizeof(float);
  float* ha = (float*)malloc(bytes);
  float* hb = (float*)malloc(bytes);
  float* hc = (float*)malloc(bytes);
  for (int i = 0; i < n; ++i) {
    ha[i] = sinf((float)i) * 3.0f;
    hb[i] = cosf((float)i) * 2.0f;
  }
  float *da, *db, *dc;
  CHECK(cudaMalloc(&da, bytes));
  CHECK(cudaMalloc((void**)&db, bytes));
  CHECK(cudaMalloc(&dc, bytes));
  CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  vecAdd<<<blocks, threads>>>(da, db, dc, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    if (fabsf(hc[i] - (ha[i] + hb[i])) > 1e-5f) {
      printf("FAIL: vecAdd mismatch at %d: %f vs %f\n", i, hc[i], ha[i] + hb[i]);
      return 1;
    }
  }

  // device pointer arithmetic: operate on the second half only
  const int half = n / 2;
  CHECK(cudaMemset(dc, 0, bytes));
  vecAdd<<<(n - half + 127) / 128, 128>>>(da + half, db + half, dc + half, n - half);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    float want = i < half ? 0.0f : ha[i] + hb[i];
    if (fabsf(hc[i] - want) > 1e-5f) {
      printf("FAIL: offset vecAdd mismatch at %d: %f vs %f\n", i, hc[i], want);
      return 1;
    }
  }

  // grid-stride loop, size_t parameter, implicit int->float conversion of an argument
  CHECK(cudaMemcpy(dc, hb, bytes, cudaMemcpyHostToDevice));
  saxpy<<<64, 128>>>(n, 2, da, dc);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    if (fabsf(hc[i] - (2.0f * ha[i] + hb[i])) > 1e-5f) {
      printf("FAIL: saxpy mismatch at %d\n", i);
      return 1;
    }
  }
  CHECK(cudaFree(da));
  CHECK(cudaFree(db));
  CHECK(cudaFree(dc));
  free(ha);
  free(hb);
  free(hc);
  printf("PASS t01_vector_add\n");
  return 0;
}
