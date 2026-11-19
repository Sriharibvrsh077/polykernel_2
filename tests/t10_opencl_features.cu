// C++ features that the OpenCL translation supports (so this test must run on OpenCL):
// default kernel arguments, device functions with reference parameters, block-level
// cooperative groups, launches through function pointers, identifiers that are OpenCL
// keywords, vector types, unrolled loops, helper structs passed by value.
#include "common.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ inline void swapIfGreater(unsigned& a, unsigned& b, unsigned dir) {
  if ((a > b) == dir) {
    unsigned t = a;
    a = b;
    b = t;
  }
}

// bitonic sort of one block (blockDim.x == 256, two elements per thread)
__global__ void bitonicBlock(unsigned* data, unsigned dir = 1) {
  cg::thread_block cta = cg::this_thread_block();
  __shared__ unsigned s[512];
  unsigned* base = data + blockIdx.x * 512;
  s[threadIdx.x] = base[threadIdx.x];
  s[threadIdx.x + 256] = base[threadIdx.x + 256];
  for (unsigned size = 2; size <= 512; size <<= 1) {
    unsigned ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);
    for (unsigned stride = size / 2; stride > 0; stride >>= 1) {
      cg::sync(cta);
      unsigned pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      swapIfGreater(s[pos], s[pos + stride], ddd);
    }
  }
  cta.sync();
  base[threadIdx.x] = s[threadIdx.x];
  base[threadIdx.x + 256] = s[threadIdx.x + 256];
}

struct Range {
  int lo, hi;
};

__global__ void countInRange(const int* v, int n, Range r, int* out) {
  cg::thread_block block = cg::this_thread_block();
  __shared__ int local;   // 'local' is a keyword in OpenCL C
  if (block.thread_rank() == 0) local = 0;
  block.sync();
  int i = blockIdx.x * block.size() + block.thread_rank();
  if (i < n && v[i] >= r.lo && v[i] < r.hi) atomicAdd(&local, 1);
  block.sync();
  if (block.thread_rank() == 0) atomicAdd(out, local);
}

__global__ void scaleA(float4* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float4 v = p[i];
    p[i] = make_float4(v.x * 2.0f, v.y * 2.0f, v.z * 2.0f, v.w * 2.0f);
  }
}
__global__ void scaleB(float4* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float4 v = p[i];
    float add = 0.0f;
#pragma unroll 4
    for (int k = 0; k < 4; ++k) add += 0.25f;
    v.x += add; v.y += add; v.z += add; v.w += add;
    p[i] = v;
  }
}

int main() {
  // sorting with default argument + reference parameters + cg::thread_block
  const int nb = 8, n = nb * 512;
  unsigned* h = new unsigned[n];
  unsigned x = 7;
  for (int i = 0; i < n; ++i) { x = x * 1664525u + 1013904223u; h[i] = x >> 8; }
  unsigned* d;
  CHECK(cudaMalloc(&d, n * sizeof(unsigned)));
  CHECK(cudaMemcpy(d, h, n * sizeof(unsigned), cudaMemcpyHostToDevice));
  bitonicBlock<<<nb, 256>>>(d);        // dir defaults to ascending
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(h, d, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
  for (int b = 0; b < nb; ++b)
    for (int i = 1; i < 512; ++i) EXPECT(h[b * 512 + i - 1] <= h[b * 512 + i], "block %d not sorted at %d", b, i);
  bitonicBlock<<<nb, 256>>>(d, 0);     // descending
  CHECK(cudaMemcpy(h, d, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
  for (int b = 0; b < nb; ++b)
    for (int i = 1; i < 512; ++i) EXPECT(h[b * 512 + i - 1] >= h[b * 512 + i], "block %d not descending", b);

  // struct by value + block-level groups
  const int m = 10000;
  int* hv = new int[m];
  int want = 0;
  for (int i = 0; i < m; ++i) { hv[i] = (i * 31) % 1000; want += hv[i] >= 100 && hv[i] < 400; }
  int *dv, *dc;
  CHECK(cudaMalloc(&dv, m * sizeof(int)));
  CHECK(cudaMalloc(&dc, sizeof(int)));
  CHECK(cudaMemcpy(dv, hv, m * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(dc, 0, sizeof(int)));
  Range r = {100, 400};
  countInRange<<<(m + 127) / 128, 128>>>(dv, m, r, dc);
  int got = 0;
  CHECK(cudaMemcpy(&got, dc, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT(got == want, "countInRange %d vs %d", got, want);

  // launches through a function pointer table
  const int nf = 777;
  float4* hf = new float4[nf];
  for (int i = 0; i < nf; ++i) hf[i] = make_float4((float)i, 1.0f, -2.0f, 0.5f);
  float4* df;
  CHECK(cudaMalloc(&df, nf * sizeof(float4)));
  CHECK(cudaMemcpy(df, hf, nf * sizeof(float4), cudaMemcpyHostToDevice));
  void (*ops[2])(float4*, int) = {scaleA, scaleB};
  for (int k = 0; k < 2; ++k) ops[k]<<<(nf + 63) / 64, 64>>>(df, nf);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(hf, df, nf * sizeof(float4), cudaMemcpyDeviceToHost));
  for (int i = 0; i < nf; ++i)
    EXPECT(hf[i].x == 2.0f * i + 1.0f && hf[i].y == 3.0f && hf[i].z == -3.0f && hf[i].w == 2.0f, "ops[%d]", i);
  printf("PASS t10_opencl_features\n");
  return 0;
}
