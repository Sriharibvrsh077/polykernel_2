// Cooperative-groups partitions and warp matching in divergent code:
// coalesced_group shfl_up/shfl_down/ballot (rank order), binary_partition, labeled_partition on a tile,
// __match_any_sync and __match_all_sync. Every result is checked against a host model of the semantics.
#include <cooperative_groups.h>
#include <cstdio>
namespace cg = cooperative_groups;

__global__ void partitions(const int* in, int* out) {
  int t = threadIdx.x + blockIdx.x * blockDim.x;
  int v = in[t];
  int* o = out + t * 8;
  if (v % 3 != 0) {   // divergent branch: about two thirds of the lanes
    cg::coalesced_group g = cg::coalesced_threads();
    o[0] = g.shfl_up(v, 2);
    o[1] = g.shfl_down(v, 1);
    o[2] = (int)g.ballot(v > 50);
    cg::coalesced_group h = cg::binary_partition(g, (v & 1) != 0);
    o[3] = (int)h.size() * 100 + (int)h.thread_rank();
  } else {
    o[0] = o[1] = o[2] = o[3] = -1;
  }
  cg::thread_block_tile<32> w = cg::tiled_partition<32>(cg::this_thread_block());
  cg::coalesced_group lp = cg::labeled_partition(w, v % 5);
  o[4] = (int)lp.size() * 100 + (int)lp.thread_rank();
  o[5] = (int)__match_any_sync(0xffffffffu, v % 7);
  int pred = 0;
  o[6] = (int)__match_all_sync(0xffffffffu, v < 1000 ? 1 : v, &pred);
  o[7] = pred;
}

static int popc(unsigned x) {
  int n = 0;
  for (; x; x &= x - 1) ++n;
  return n;
}
static int lane_of_rank(unsigned mask, int r) {
  for (int i = 0; i < 32; ++i)
    if ((mask >> i) & 1u) {
      if (r == 0) return i;
      --r;
    }
  return -1;
}

int main() {
  const int n = 256;
  static int h[n], r[n * 8], want[n * 8];
  for (int i = 0; i < n; ++i) h[i] = (i * 37 + 11) % 97;
  int *din, *dout;
  cudaMalloc(&din, sizeof h);
  cudaMalloc(&dout, sizeof r);
  cudaMemcpy(din, h, sizeof h, cudaMemcpyHostToDevice);
  partitions<<<n / 64, 64>>>(din, dout);
  cudaError_t err = cudaDeviceSynchronize();
  cudaMemcpy(r, dout, sizeof r, cudaMemcpyDeviceToHost);
  // host model, warp by warp
  for (int w0 = 0; w0 < n; w0 += 32) {
    const int* v = h + w0;
    unsigned g = 0;
    for (int l = 0; l < 32; ++l)
      if (v[l] % 3 != 0) g |= 1u << l;
    for (int l = 0; l < 32; ++l) {
      int* o = want + (w0 + l) * 8;
      if ((g >> l) & 1u) {
        int rank = popc(g & ((1u << l) - 1u)), size = popc(g);
        o[0] = rank >= 2 ? v[lane_of_rank(g, rank - 2)] : v[l];
        o[1] = rank + 1 < size ? v[lane_of_rank(g, rank + 1)] : v[l];
        unsigned b = 0;
        for (int k = 0; k < size; ++k)
          if (v[lane_of_rank(g, k)] > 50) b |= 1u << k;
        o[2] = (int)b;
        unsigned hm = 0;
        for (int k = 0; k < 32; ++k)
          if (((g >> k) & 1u) && (v[k] & 1) == (v[l] & 1)) hm |= 1u << k;
        o[3] = popc(hm) * 100 + popc(hm & ((1u << l) - 1u));
      } else {
        o[0] = o[1] = o[2] = o[3] = -1;
      }
      unsigned lm = 0, mm = 0;
      for (int k = 0; k < 32; ++k) {
        if (v[k] % 5 == v[l] % 5) lm |= 1u << k;
        if (v[k] % 7 == v[l] % 7) mm |= 1u << k;
      }
      o[4] = popc(lm) * 100 + popc(lm & ((1u << l) - 1u));
      o[5] = (int)mm;
      o[6] = -1;   // all lanes pass the same value: the full mask
      o[7] = 1;
    }
  }
  int bad = 0;
  for (int i = 0; i < n * 8; ++i)
    if (r[i] != want[i]) {
      if (++bad <= 5) printf("mismatch: thread %d field %d: got %d, want %d\n", i / 8, i % 8, r[i], want[i]);
    }
  bool ok = err == cudaSuccess && bad == 0;
  printf("%s t13_cg_partitions: %d of %d values match\n", ok ? "PASS" : "FAIL", n * 8 - bad, n * 8);
  cudaFree(din);
  cudaFree(dout);
  return ok ? 0 : 1;
}
