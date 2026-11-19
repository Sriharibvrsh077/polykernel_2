// Warp intrinsics and cooperative groups: shuffles, votes, __syncwarp, partial warps,
// __syncthreads_count, tiled_partition + cg::reduce, block-wide reduction with warp shuffles.
#include "common.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__inline__ __device__ float warpReduceSum(float v) {
  for (int off = warpSize / 2; off > 0; off /= 2) v += __shfl_down_sync(0xffffffff, v, off);
  return v;
}

__global__ void blockReduce(const float* in, float* out, int n) {
  __shared__ float warpSums[32];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = i < n ? in[i] : 0.0f;
  v = warpReduceSum(v);
  int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
  if (lane == 0) warpSums[wid] = v;
  __syncthreads();
  int nw = (blockDim.x + warpSize - 1) / warpSize;
  v = (threadIdx.x < nw) ? warpSums[lane] : 0.0f;
  if (wid == 0) v = warpReduceSum(v);
  if (threadIdx.x == 0) atomicAdd(out, v);
}

__global__ void votes(const int* in, unsigned* ballots, int* anyAll, int* counts) {
  int v = in[blockIdx.x * blockDim.x + threadIdx.x];
  unsigned b = __ballot_sync(0xffffffff, v > 0);
  int a = __any_sync(0xffffffff, v > 5);
  int l = __all_sync(0xffffffff, v >= 0);
  int c = __syncthreads_count(v > 0);
  int lane = threadIdx.x & 31;
  if (lane == 0) {
    int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    ballots[w] = b;
    anyAll[2 * w] = a;
    anyAll[2 * w + 1] = l;
  }
  if (threadIdx.x == 0) counts[blockIdx.x] = c;
}

__global__ void shuffles(int* out) {
  int lane = threadIdx.x & 31;
  int bcast = __shfl_sync(0xffffffff, lane * 10, 3);
  int up = __shfl_up_sync(0xffffffff, lane, 1);
  int x = __shfl_xor_sync(0xffffffff, lane, 1);
  int seg = __shfl_sync(0xffffffff, lane, 0, 8);   // width-8 segments
  __syncwarp();
  out[threadIdx.x * 4 + 0] = bcast;
  out[threadIdx.x * 4 + 1] = up;
  out[threadIdx.x * 4 + 2] = x;
  out[threadIdx.x * 4 + 3] = seg;
}

__global__ void cgReduce(const int* in, int* out) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<16> tile = cg::tiled_partition<16>(block);
  int v = in[block.group_index().x * block.size() + block.thread_rank()];
  int s = cg::reduce(tile, v, cg::plus<int>());
  block.sync();
  if (tile.thread_rank() == 0) atomicAdd(out, s);
}

int main() {
  // block reduction with a partial last warp (blockDim 48)
  const int n = 10000;
  float* h = new float[n];
  double ref = 0;
  for (int i = 0; i < n; ++i) { h[i] = (float)(i % 7) - 2.5f; ref += h[i]; }
  float *din, *dout;
  CHECK(cudaMalloc(&din, n * sizeof(float)));
  CHECK(cudaMalloc(&dout, sizeof(float)));
  CHECK(cudaMemcpy(din, h, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(dout, 0, sizeof(float)));
  blockReduce<<<(n + 47) / 48, 48>>>(din, dout, n);
  CHECK(cudaGetLastError());
  float got = 0;
  CHECK(cudaMemcpy(&got, dout, sizeof(float), cudaMemcpyDeviceToHost));
  EXPECT(fabs(got - ref) < 1e-2, "blockReduce %f vs %f", got, ref);

  // votes
  const int nt = 128;
  int hin[nt];
  for (int i = 0; i < nt; ++i) hin[i] = (i % 3 == 0) ? (i % 9) : -(i % 2);
  int *dvin, *danyall, *dcounts;
  unsigned* dball;
  CHECK(cudaMalloc(&dvin, sizeof hin));
  CHECK(cudaMalloc(&dball, 4 * sizeof(unsigned)));
  CHECK(cudaMalloc(&danyall, 8 * sizeof(int)));
  CHECK(cudaMalloc(&dcounts, 2 * sizeof(int)));
  CHECK(cudaMemcpy(dvin, hin, sizeof hin, cudaMemcpyHostToDevice));
  votes<<<2, 64>>>(dvin, dball, danyall, dcounts);
  CHECK(cudaGetLastError());
  unsigned hb[4];
  int ha[8], hc[2];
  CHECK(cudaMemcpy(hb, dball, sizeof hb, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(ha, danyall, sizeof ha, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hc, dcounts, sizeof hc, cudaMemcpyDeviceToHost));
  for (int w = 0; w < 4; ++w) {
    unsigned eb = 0;
    int ea = 0, el = 1;
    for (int l = 0; l < 32; ++l) {
      int v = hin[w * 32 + l];
      if (v > 0) eb |= 1u << l;
      if (v > 5) ea = 1;
      if (v < 0) el = 0;
    }
    EXPECT(hb[w] == eb, "ballot warp %d: %08x vs %08x", w, hb[w], eb);
    EXPECT(ha[2 * w] == ea && ha[2 * w + 1] == el, "any/all warp %d", w);
  }
  for (int b = 0; b < 2; ++b) {
    int ec = 0;
    for (int i = 0; i < 64; ++i) ec += hin[b * 64 + i] > 0;
    EXPECT(hc[b] == ec, "__syncthreads_count block %d: %d vs %d", b, hc[b], ec);
  }

  // shuffles
  int* dsh;
  CHECK(cudaMalloc(&dsh, 64 * 4 * sizeof(int)));
  shuffles<<<1, 64>>>(dsh);
  int hs[64 * 4];
  CHECK(cudaMemcpy(hs, dsh, sizeof hs, cudaMemcpyDeviceToHost));
  for (int t = 0; t < 64; ++t) {
    int lane = t & 31;
    EXPECT(hs[t * 4 + 0] == 30, "shfl bcast %d", t);
    EXPECT(hs[t * 4 + 1] == (lane == 0 ? 0 : lane - 1), "shfl_up %d = %d", t, hs[t * 4 + 1]);
    EXPECT(hs[t * 4 + 2] == (lane ^ 1), "shfl_xor %d", t);
    EXPECT(hs[t * 4 + 3] == (lane & ~7), "shfl width 8 %d = %d", t, hs[t * 4 + 3]);
  }

  // cooperative groups
  const int nc = 256;
  int hcg[nc], cref = 0;
  for (int i = 0; i < nc; ++i) { hcg[i] = i % 13; cref += hcg[i]; }
  int *dcg, *dcgout;
  CHECK(cudaMalloc(&dcg, sizeof hcg));
  CHECK(cudaMalloc(&dcgout, sizeof(int)));
  CHECK(cudaMemcpy(dcg, hcg, sizeof hcg, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(dcgout, 0, sizeof(int)));
  cgReduce<<<nc / 64, 64>>>(dcg, dcgout);
  CHECK(cudaGetLastError());
  int cgot = 0;
  CHECK(cudaMemcpy(&cgot, dcgout, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT(cgot == cref, "cg::reduce %d vs %d", cgot, cref);
  printf("PASS t08_warp_coop\n");
  return 0;
}
