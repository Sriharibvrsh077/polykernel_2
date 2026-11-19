// Tree reduction with dynamic shared memory (extern __shared__), float and int atomics.
#include "common.h"

__global__ void blockSum(const float* in, float* partial, float* total, int n) {
  extern __shared__ float sdata[];
  unsigned tid = threadIdx.x;
  unsigned i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  float v = 0.0f;
  if (i < n) v += in[i];
  if (i + blockDim.x < n) v += in[i + blockDim.x];
  sdata[tid] = v;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) {
    partial[blockIdx.x] = sdata[0];
    atomicAdd(total, sdata[0]);
  }
}

__global__ void countPositive(const int* in, int n, int* count, unsigned int* maxv) {
  __shared__ int local;
  if (threadIdx.x == 0) local = 0;
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && in[i] > 0) atomicAdd(&local, 1);
  if (i < n) atomicMax(maxv, (unsigned int)in[i]);
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(count, local);
}

int main() {
  const int n = 1 << 18;
  const int threads = 256;
  const int blocks = (n + threads * 2 - 1) / (threads * 2);
  float* h = new float[n];
  double ref = 0.0;
  for (int i = 0; i < n; ++i) {
    h[i] = (float)(i % 17) * 0.25f;
    ref += h[i];
  }
  float *din, *dpart, *dtot;
  CHECK(cudaMalloc(&din, n * sizeof(float)));
  CHECK(cudaMalloc(&dpart, blocks * sizeof(float)));
  CHECK(cudaMalloc(&dtot, sizeof(float)));
  CHECK(cudaMemcpy(din, h, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(dtot, 0, sizeof(float)));
  blockSum<<<blocks, threads, threads * sizeof(float)>>>(din, dpart, dtot, n);
  CHECK(cudaGetLastError());
  float* part = new float[blocks];
  float tot = 0.0f;
  CHECK(cudaMemcpy(part, dpart, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&tot, dtot, sizeof(float), cudaMemcpyDeviceToHost));
  double s = 0.0;
  for (int b = 0; b < blocks; ++b) s += part[b];
  EXPECT(fabs(s - ref) < 1e-6 * ref + 1e-3, "partial sums %f vs %f", s, ref);
  EXPECT(fabs(tot - ref) < 1e-4 * ref, "atomic total %f vs %f", tot, ref);

  int* hi = new int[n];
  int pos = 0;
  unsigned mx = 0;
  for (int i = 0; i < n; ++i) {
    hi[i] = (i * 37) % 101 - 50;
    if (hi[i] > 0) ++pos;
    if ((unsigned)hi[i] > mx) mx = (unsigned)hi[i];
  }
  int *dint, *dcount;
  unsigned* dmax;
  CHECK(cudaMalloc(&dint, n * sizeof(int)));
  CHECK(cudaMalloc(&dcount, sizeof(int)));
  CHECK(cudaMalloc(&dmax, sizeof(unsigned)));
  CHECK(cudaMemcpy(dint, hi, n * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(dcount, 0, sizeof(int)));
  CHECK(cudaMemset(dmax, 0, sizeof(unsigned)));
  countPositive<<<(n + 127) / 128, 128>>>(dint, n, dcount, dmax);
  CHECK(cudaGetLastError());
  int gpos = 0;
  unsigned gmax = 0;
  CHECK(cudaMemcpy(&gpos, dcount, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&gmax, dmax, sizeof(unsigned), cudaMemcpyDeviceToHost));
  EXPECT(gpos == pos, "positive count %d vs %d", gpos, pos);
  EXPECT(gmax == mx, "max %u vs %u", gmax, mx);
  printf("PASS t03_reduction\n");
  return 0;
}
