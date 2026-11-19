// Converged-lane groups in divergent code: coalesced_threads() (warp-aggregated atomics),
// labeled_partition, and __activemask() in both branches of an if/else.
#include <cstdio>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__device__ int atomicAggInc(int* counter) {
  cg::coalesced_group active = cg::coalesced_threads();
  int res = 0;
  if (active.thread_rank() == 0) res = atomicAdd(counter, (int)active.size());
  res = active.shfl(res, 0);
  return res + (int)active.thread_rank();
}
__global__ void filter(int* dst, int* nres, const int* src, int n) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < n && src[id] > 0) dst[atomicAggInc(nres)] = src[id];
}
__global__ void ballotActive(const int* src, unsigned* out) {
  int v = src[threadIdx.x];
  if (v > 0) { unsigned m = __activemask(); out[threadIdx.x] = __ballot_sync(m, v > 5); }
  else { unsigned m = __activemask(); out[threadIdx.x] = m; }
}
__global__ void bucketCount(const int* src, int n, int* counts) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < n && src[id] >= 0) {
    cg::coalesced_group active = cg::coalesced_threads();
    int bucket = src[id] % 4;
    cg::coalesced_group same = cg::labeled_partition(active, bucket);
    if (same.thread_rank() == 0) atomicAdd(&counts[bucket], (int)same.size());
  }
}
int main() {
  const int n = 100000;
  int* h = new int[n]; int want = 0;
  for (int i = 0; i < n; ++i) { h[i] = (i * 7919) % 13 - 4; want += h[i] > 0; }
  int *src, *dst, *cnt; cudaMalloc(&src, n * 4); cudaMalloc(&dst, n * 4); cudaMalloc(&cnt, 4);
  cudaMemcpy(src, h, n * 4, cudaMemcpyHostToDevice); cudaMemset(cnt, 0, 4); cudaMemset(dst, 0, n * 4);
  filter<<<(n + 255) / 256, 256>>>(dst, cnt, src, n);
  int got = 0; cudaMemcpy(&got, cnt, 4, cudaMemcpyDeviceToHost);
  int* hd = new int[n]; cudaMemcpy(hd, dst, n * 4, cudaMemcpyDeviceToHost);
  long long s1 = 0, s2 = 0; for (int i = 0; i < n; ++i) { if (h[i] > 0) s1 += h[i]; if (i < got) s2 += hd[i]; }
  unsigned* out; cudaMalloc(&out, 32 * 4); ballotActive<<<1, 32>>>(src, out); unsigned ho[32]; cudaMemcpy(ho, out, sizeof ho, cudaMemcpyDeviceToHost);
  unsigned pos = 0, big = 0; for (int i = 0; i < 32; ++i) { if (h[i] > 0) pos |= 1u << i; if (h[i] > 5) big |= 1u << i; }
  int* bc; cudaMalloc(&bc, 4 * 4); cudaMemset(bc, 0, 16);
  bucketCount<<<(n + 127) / 128, 128>>>(src, n, bc);
  int hb[4]; cudaMemcpy(hb, bc, sizeof hb, cudaMemcpyDeviceToHost);
  int wb[4] = {0, 0, 0, 0}; for (int i = 0; i < n; ++i) if (h[i] >= 0) wb[h[i] % 4]++;
  int ok = got == want && s1 == s2;
  for (int b = 0; b < 4; ++b) ok &= hb[b] == wb[b];
  for (int i = 0; i < 32; ++i) ok &= (h[i] > 0) ? ho[i] == big : ho[i] == ~pos;
  printf("%s: %d of %d kept, checksum %lld vs %lld\n", ok ? "PASS t11_coalesced_groups" : "FAIL t11_coalesced_groups", got, want, s2, s1);
  return !ok;
}
