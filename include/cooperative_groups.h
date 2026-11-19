// PolyKernel: cooperative groups (block, warp tiles, coalesced groups, reduce/scan)
// for the CPU execution model. Grid-wide sync is not supported.
#ifndef POLYKERNEL_COOPERATIVE_GROUPS_H
#define POLYKERNEL_COOPERATIVE_GROUPS_H
#include "cuda_runtime.h"

namespace cooperative_groups {
namespace details {
inline unsigned block_threads() {
  const dim3& b = ::pk::cpu::tl_block->bdim;
  return b.x * b.y * b.z;
}
inline unsigned linear_tid() { return ::pk::cpu::tl_lane->linear; }
}  // namespace details

// Type-erased group: a block, a warp tile (size <= 32) or a coalesced set of lanes.
class thread_group {
 public:
  enum : unsigned { kBlock = 1, kGrid = 2, kTile = 3, kCoalesced = 4 };
  explicit thread_group(unsigned kind = kBlock, unsigned tile = 0, unsigned mask = 0)
      : kind_(kind), tile_(tile), mask_(mask) {}
  unsigned lane_mask() const {
    unsigned lane = details::linear_tid() & 31u;
    if (kind_ == kCoalesced) return mask_;
    if (kind_ == kTile && tile_ < 32) return ((1u << tile_) - 1u) << (lane & ~(tile_ - 1u));
    return 0xffffffffu;
  }
  void sync() const {
    if (kind_ == kBlock) __syncthreads();
    else ::pk::cpu::warp_sync(lane_mask());
  }
  unsigned long long thread_rank() const {
    if (kind_ == kBlock) return details::linear_tid();
    if (kind_ == kTile) return details::linear_tid() & (tile_ - 1u);
    unsigned lane = details::linear_tid() & 31u;
    return (unsigned)__builtin_popcount(mask_ & ((1u << lane) - 1u));
  }
  unsigned long long size() const {
    if (kind_ == kBlock) return details::block_threads();
    if (kind_ == kTile) return tile_;
    return (unsigned)__builtin_popcount(mask_);
  }
  unsigned long long num_threads() const { return size(); }
  bool is_valid() const { return true; }
  unsigned get_type() const { return kind_; }

 protected:
  unsigned kind_, tile_, mask_;
};

class thread_block : public thread_group {
 public:
  thread_block() : thread_group(kBlock) {}
  static void sync() { __syncthreads(); }
  static void barrier_wait() { __syncthreads(); }
  static unsigned thread_rank() { return details::linear_tid(); }
  static unsigned size() { return details::block_threads(); }
  static unsigned num_threads() { return details::block_threads(); }
  static dim3 group_index() { return dim3(blockIdx.x, blockIdx.y, blockIdx.z); }
  static dim3 thread_index() { return dim3(threadIdx.x, threadIdx.y, threadIdx.z); }
  static dim3 group_dim() { return blockDim; }
  static dim3 dim_threads() { return blockDim; }
};
inline thread_block this_thread_block() { return thread_block(); }

class grid_group : public thread_group {
 public:
  grid_group() : thread_group(kGrid) {}
  static void sync() { ::pk::cpu::grid_sync_unsupported(); }
  static bool is_valid() { return false; }
  static unsigned long long thread_rank() {
    unsigned long long b = blockIdx.x + (unsigned long long)gridDim.x * (blockIdx.y + (unsigned long long)gridDim.y * blockIdx.z);
    return b * details::block_threads() + details::linear_tid();
  }
  static unsigned long long size() {
    return (unsigned long long)gridDim.x * gridDim.y * gridDim.z * details::block_threads();
  }
  static unsigned long long num_threads() { return size(); }
  static dim3 group_dim() { return gridDim; }
  static dim3 block_index() { return dim3(blockIdx.x, blockIdx.y, blockIdx.z); }
};
inline grid_group this_grid() { return grid_group(); }

template <unsigned N>
class thread_block_tile : public thread_group {
  static_assert(N >= 1 && N <= 32 && (N & (N - 1)) == 0, "tile size must be a power of two <= 32");

 public:
  thread_block_tile() : thread_group(kTile, N) {}
  static unsigned mask() {
    unsigned lane = details::linear_tid() & 31u;
    const unsigned bits = 0xffffffffu >> (32u - N);   // N low bits, no shift by 32 when N == 32
    return N == 32 ? bits : bits << (lane & ~(N - 1u));
  }
  static void sync() { ::pk::cpu::warp_sync(mask()); }
  static unsigned thread_rank() { return details::linear_tid() & (N - 1u); }
  static unsigned size() { return N; }
  static unsigned num_threads() { return N; }
  static unsigned meta_group_rank() { return details::linear_tid() / N; }
  static unsigned meta_group_size() { return (details::block_threads() + N - 1) / N; }
  template <class T> static T shfl(T v, int src) { return __shfl_sync(mask(), v, src, N); }
  template <class T> static T shfl_down(T v, unsigned d) { return __shfl_down_sync(mask(), v, d, N); }
  template <class T> static T shfl_up(T v, unsigned d) { return __shfl_up_sync(mask(), v, d, N); }
  template <class T> static T shfl_xor(T v, int m) { return __shfl_xor_sync(mask(), v, m, N); }
  static int any(int p) { return __any_sync(mask(), p); }
  static int all(int p) { return __all_sync(mask(), p); }
  static unsigned ballot(int p) {
    unsigned lane = details::linear_tid() & 31u;
    unsigned shift = (N == 32) ? 0u : (lane & ~(N - 1u));
    unsigned b = __ballot_sync(mask(), p) >> shift;
    return N == 32 ? b : (b & ((1u << N) - 1u));
  }
  template <class T> static unsigned match_any(T v) { return __match_any_sync(mask(), v); }
};

template <unsigned N, class Parent>
inline thread_block_tile<N> tiled_partition(const Parent&) { return thread_block_tile<N>(); }
inline thread_group tiled_partition(const thread_group&, unsigned tile_size) {
  if (tile_size == 0 || tile_size > 32 || (tile_size & (tile_size - 1)) != 0) {
    fprintf(stderr, "polykernel: tiled_partition size must be a power of two <= 32\n");
    abort();
  }
  return thread_group(thread_group::kTile, tile_size);
}

class coalesced_group : public thread_group {
 public:
  __attribute__((always_inline)) coalesced_group() : thread_group(kCoalesced, 0, __activemask()) {}
  explicit coalesced_group(unsigned lane_mask) : thread_group(kCoalesced, 0, lane_mask) {}
  void sync() const { ::pk::cpu::warp_sync(mask_); }
  unsigned size() const { return (unsigned)__builtin_popcount(mask_); }
  unsigned num_threads() const { return size(); }
  unsigned thread_rank() const {
    unsigned lane = details::linear_tid() & 31u;
    return (unsigned)__builtin_popcount(mask_ & ((1u << lane) - 1u));
  }
  unsigned lane_of_rank(unsigned r) const {
    unsigned m = mask_;
    for (unsigned i = 0; i < 32; ++i)
      if ((m >> i) & 1u) {
        if (r == 0) return i;
        --r;
      }
    return 0;
  }
  template <class T> T shfl(T v, int rank) const { return __shfl_sync(mask_, v, (int)lane_of_rank((unsigned)rank)); }
  template <class T> T shfl_down(T v, unsigned d) const {
    unsigned r = thread_rank() + d;
    unsigned src = r < size() ? lane_of_rank(r) : (details::linear_tid() & 31u);
    return __shfl_sync(mask_, v, (int)src);
  }
  template <class T> T shfl_up(T v, unsigned d) const {
    unsigned r = thread_rank();
    unsigned src = r >= d ? lane_of_rank(r - d) : (details::linear_tid() & 31u);
    return __shfl_sync(mask_, v, (int)src);
  }
  int any(int p) const { return __any_sync(mask_, p); }
  int all(int p) const { return __all_sync(mask_, p); }
  unsigned ballot(int p) const {
    unsigned b = __ballot_sync(mask_, p), r = 0, k = 0;
    for (unsigned i = 0; i < 32; ++i)
      if ((mask_ >> i) & 1u) { if ((b >> i) & 1u) r |= 1u << k; ++k; }
    return r;
  }
};
__attribute__((always_inline)) inline coalesced_group coalesced_threads() { return coalesced_group(); }

// Split a group into subgroups of lanes with equal labels.
template <class Label>
inline coalesced_group labeled_partition(const coalesced_group& g, Label label) {
  return coalesced_group(__match_any_sync(g.lane_mask(), label));
}
template <unsigned N, class Label>
inline coalesced_group labeled_partition(const thread_block_tile<N>& g, Label label) {
  return coalesced_group(__match_any_sync(g.mask(), label));
}
inline coalesced_group binary_partition(const coalesced_group& g, bool pred) { return labeled_partition(g, (int)pred); }
template <unsigned N>
inline coalesced_group binary_partition(const thread_block_tile<N>& g, bool pred) {
  return labeled_partition(g, (int)pred);
}

template <class G> inline void sync(const G& g) { g.sync(); }

// Functors used with reduce / scan
template <class T> struct plus { T operator()(const T& a, const T& b) const { return a + b; } };
template <class T> struct less { T operator()(const T& a, const T& b) const { return a < b ? a : b; } };
template <class T> struct greater { T operator()(const T& a, const T& b) const { return a > b ? a : b; } };
template <class T> struct bit_and { T operator()(const T& a, const T& b) const { return a & b; } };
template <class T> struct bit_or { T operator()(const T& a, const T& b) const { return a | b; } };
template <class T> struct bit_xor { T operator()(const T& a, const T& b) const { return a ^ b; } };

template <unsigned N, class T, class Op>
inline T reduce(const thread_block_tile<N>& g, T val, Op op) {
  for (unsigned off = N / 2; off > 0; off /= 2) val = op(val, g.shfl_xor(val, (int)off));
  return val;
}
template <class T, class Op>
inline T reduce(const coalesced_group& g, T val, Op op) {
  // gather to rank 0 through shuffles, then broadcast
  unsigned n = g.size();
  for (unsigned off = 1; off < n; off *= 2) {
    T other = g.shfl_down(val, off);
    if (g.thread_rank() + off < n && (g.thread_rank() % (2 * off)) == 0) val = op(val, other);
  }
  return g.shfl(val, 0);
}
template <unsigned N, class T, class Op>
inline T inclusive_scan(const thread_block_tile<N>& g, T val, Op op) {
  for (unsigned off = 1; off < N; off *= 2) {
    T other = g.shfl_up(val, off);
    if (g.thread_rank() >= off) val = op(val, other);
  }
  return val;
}
template <unsigned N, class T>
inline T inclusive_scan(const thread_block_tile<N>& g, T val) { return inclusive_scan(g, val, plus<T>()); }
template <unsigned N, class T, class Op>
inline T exclusive_scan(const thread_block_tile<N>& g, T val, Op op) {
  T inc = inclusive_scan(g, val, op);
  T prev = g.shfl_up(inc, 1);
  return g.thread_rank() == 0 ? T() : prev;
}
template <unsigned N, class T>
inline T exclusive_scan(const thread_block_tile<N>& g, T val) { return exclusive_scan(g, val, plus<T>()); }

}  // namespace cooperative_groups

#endif  // POLYKERNEL_COOPERATIVE_GROUPS_H
