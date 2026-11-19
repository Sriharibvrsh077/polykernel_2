// The smallest useful CUDA program: build it with
//   bin/pkcc -o hello examples/hello.cu && ./hello
// then try   PK_VERBOSE=1 ./hello             (shows which back end ran it)
// and        bin/pkcc --target=aarch64-linux-gnu -o hello-arm64 examples/hello.cu
#include <cstdio>

__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 20;
  float *a, *b, *c;
  cudaMallocManaged(&a, n * sizeof(float));
  cudaMallocManaged(&b, n * sizeof(float));
  cudaMallocManaged(&c, n * sizeof(float));
  for (int i = 0; i < n; ++i) { a[i] = 1.0f; b[i] = 2.0f; }

  add<<<(n + 255) / 256, 256>>>(a, b, c, n);
  cudaDeviceSynchronize();

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("ran on: %s\nc[0] = %.1f, c[n-1] = %.1f\n", prop.name, c[0], c[n - 1]);
  int ok = c[0] == 3.0f && c[n - 1] == 3.0f;
  cudaFree(a); cudaFree(b); cudaFree(c);
  return ok ? 0 : 1;
}
