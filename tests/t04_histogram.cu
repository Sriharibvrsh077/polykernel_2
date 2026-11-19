// Shared-memory histogram with atomics on __shared__ and global memory; 2D grid of 1D blocks.
#include "common.h"
#define BINS 64

__global__ void histogram(const unsigned char* data, size_t n, unsigned int* bins) {
  __shared__ unsigned int local[BINS];
  for (int b = threadIdx.x; b < BINS; b += blockDim.x) local[b] = 0;
  __syncthreads();
  size_t start = ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * gridDim.y * blockDim.x;
  for (size_t i = start; i < n; i += stride) atomicAdd(&local[data[i] % BINS], 1u);
  __syncthreads();
  for (int b = threadIdx.x; b < BINS; b += blockDim.x) atomicAdd(&bins[b], local[b]);
}

int main() {
  const size_t n = 3000017;
  unsigned char* h = (unsigned char*)malloc(n);
  unsigned ref[BINS] = {0};
  unsigned x = 12345;
  for (size_t i = 0; i < n; ++i) {
    x = x * 1103515245u + 12345u;
    h[i] = (unsigned char)(x >> 16);
    ref[h[i] % BINS]++;
  }
  unsigned char* d;
  unsigned* dbins;
  CHECK(cudaMalloc(&d, n));
  CHECK(cudaMalloc(&dbins, BINS * sizeof(unsigned)));
  CHECK(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(dbins, 0, BINS * sizeof(unsigned)));
  histogram<<<dim3(16, 8), 192>>>(d, n, dbins);
  CHECK(cudaGetLastError());
  unsigned got[BINS];
  CHECK(cudaMemcpy(got, dbins, sizeof got, cudaMemcpyDeviceToHost));
  for (int b = 0; b < BINS; ++b) EXPECT(got[b] == ref[b], "bin %d: %u vs %u", b, got[b], ref[b]);
  printf("PASS t04_histogram\n");
  return 0;
}
