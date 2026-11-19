// PolyKernel: device-side built-ins for the CPU execution model.
// Each CUDA block runs on one worker thread; each CUDA thread is a lightweight
// fiber (only when the kernel uses barriers / warp intrinsics, otherwise a loop).
#ifndef POLYKERNEL_PK_DEVICE_H
#define POLYKERNEL_PK_DEVICE_H

#include "pk_types.h"
#include <atomic>
#include <chrono>

namespace pk {
namespace cpu {
struct Lane {
  uint3 tid;
  unsigned linear;
};
struct Block {
  uint3 bid;
  dim3 bdim;
  dim3 gdim;
  void* dyn_smem;
  unsigned nthreads;
};
extern __thread Lane* tl_lane;
extern __thread Block* tl_block;
inline void* dyn_smem() { return tl_block->dyn_smem; }
void syncthreads();
int syncthreads_count(int pred);
void warp_sync(unsigned mask);
unsigned active_mask();      // lanes of the warp that have not exited
unsigned converged_here();   // lanes that reach the calling site together (see pk_cpu.cpp)
uint64_t* warp_slots();
[[noreturn]] void grid_sync_unsupported();
}  // namespace cpu
}  // namespace pk

// ---------------------------------------------------------------------------
// Built-in variables
// ---------------------------------------------------------------------------
#define threadIdx (static_cast<const ::uint3&>(::pk::cpu::tl_lane->tid))
#define blockIdx (static_cast<const ::uint3&>(::pk::cpu::tl_block->bid))
#define blockDim (static_cast<const ::dim3&>(::pk::cpu::tl_block->bdim))
#define gridDim (static_cast<const ::dim3&>(::pk::cpu::tl_block->gdim))
static constexpr int warpSize = 32;

// ---------------------------------------------------------------------------
// Synchronization
// ---------------------------------------------------------------------------
static inline void __syncthreads() { ::pk::cpu::syncthreads(); }
static inline int __syncthreads_count(int p) { return ::pk::cpu::syncthreads_count(p); }
static inline int __syncthreads_and(int p) { return ::pk::cpu::syncthreads_count(!p) == 0; }
static inline int __syncthreads_or(int p) { return ::pk::cpu::syncthreads_count(p) != 0; }
static inline void __syncwarp(unsigned mask = 0xffffffffu) { ::pk::cpu::warp_sync(mask); }
// A macro so that every call site is told apart (converged_here uses its return address).
#define __activemask() (::pk::cpu::converged_here())
static inline void __threadfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
static inline void __threadfence_block() { std::atomic_thread_fence(std::memory_order_seq_cst); }
static inline void __threadfence_system() { std::atomic_thread_fence(std::memory_order_seq_cst); }

// ---------------------------------------------------------------------------
// Warp-level primitives
// ---------------------------------------------------------------------------
namespace pk {
namespace cpu {
inline unsigned lane_id() { return tl_lane->linear & 31u; }
template <class T>
inline T shfl_from(unsigned mask, T var, unsigned src_of_lane, bool use_own) {
  static_assert(std::is_trivially_copyable<T>::value, "warp shuffles need trivially copyable types");
  unsigned lane = lane_id();
  uint64_t* s = warp_slots();
  T r = var;
  for (size_t off = 0; off < sizeof(T); off += 8) {   // 8 bytes per exchange
    size_t n = sizeof(T) - off < 8 ? sizeof(T) - off : 8;
    std::memcpy(&s[lane], reinterpret_cast<const char*>(&var) + off, n);
    warp_sync(mask);
    if (!use_own) std::memcpy(reinterpret_cast<char*>(&r) + off, &s[src_of_lane & 31u], n);
    warp_sync(mask);
  }
  return r;
}
}  // namespace cpu
}  // namespace pk

template <class T>
static inline T __shfl_sync(unsigned mask, T var, int srcLane, int width = 32) {
  unsigned lane = ::pk::cpu::lane_id();
  unsigned base = lane & ~(unsigned)(width - 1);
  return ::pk::cpu::shfl_from(mask, var, base + ((unsigned)srcLane & (unsigned)(width - 1)), false);
}
template <class T>
static inline T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = 32) {
  unsigned lane = ::pk::cpu::lane_id();
  unsigned base = lane & ~(unsigned)(width - 1);
  bool own = (lane - base) < delta;
  return ::pk::cpu::shfl_from(mask, var, lane - delta, own);
}
template <class T>
static inline T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = 32) {
  unsigned lane = ::pk::cpu::lane_id();
  unsigned base = lane & ~(unsigned)(width - 1);
  bool own = (lane - base) + delta >= (unsigned)width;
  return ::pk::cpu::shfl_from(mask, var, lane + delta, own);
}
template <class T>
static inline T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = 32) {
  unsigned lane = ::pk::cpu::lane_id();
  unsigned base = lane & ~(unsigned)(width - 1);
  unsigned src = lane ^ (unsigned)laneMask;
  bool own = src >= base + (unsigned)width;
  return ::pk::cpu::shfl_from(mask, var, src, own);
}
// Legacy (pre-CUDA 9) forms: whole warp
template <class T>
static inline T __shfl(T var, int srcLane, int width = 32) { return __shfl_sync(0xffffffffu, var, srcLane, width); }
template <class T>
static inline T __shfl_up(T var, unsigned d, int width = 32) { return __shfl_up_sync(0xffffffffu, var, d, width); }
template <class T>
static inline T __shfl_down(T var, unsigned d, int width = 32) { return __shfl_down_sync(0xffffffffu, var, d, width); }
template <class T>
static inline T __shfl_xor(T var, int m, int width = 32) { return __shfl_xor_sync(0xffffffffu, var, m, width); }

namespace pk {
namespace cpu {
// Ballot over the lanes in `mask`; also returns the lanes that took part (sampled while
// every participant is inside the vote, so lanes that exit afterwards do not matter).
inline unsigned vote(unsigned mask, int pred, unsigned* participants) {
  unsigned lane = lane_id();
  uint64_t* s = warp_slots();
  s[lane] = pred ? 1u : 0u;
  warp_sync(mask);
  unsigned live = active_mask() & (mask | (1u << lane)), r = 0;
  for (unsigned i = 0; i < 32; ++i)
    if (((live >> i) & 1u) && s[i]) r |= 1u << i;
  warp_sync(mask);
  if (participants) *participants = live;
  return r;
}
}  // namespace cpu
}  // namespace pk
static inline unsigned __ballot_sync(unsigned mask, int pred) { return ::pk::cpu::vote(mask, pred, nullptr); }
static inline int __any_sync(unsigned mask, int pred) { return ::pk::cpu::vote(mask, pred, nullptr) != 0; }
static inline int __all_sync(unsigned mask, int pred) {
  unsigned live;
  unsigned b = ::pk::cpu::vote(mask, pred, &live);
  return b == live;
}
static inline int __uni_sync(unsigned mask, int pred) {
  unsigned live;
  unsigned b = ::pk::cpu::vote(mask, pred, &live);
  return b == 0 || b == live;
}
static inline unsigned __ballot(int pred) { return __ballot_sync(0xffffffffu, pred); }
static inline int __any(int pred) { return __any_sync(0xffffffffu, pred); }
static inline int __all(int pred) { return __all_sync(0xffffffffu, pred); }

namespace pk {
namespace cpu {
template <class T>
inline unsigned match(unsigned mask, T value, unsigned* participants) {
  unsigned lane = lane_id();
  uint64_t* s = warp_slots();
  uint64_t v = 0;
  std::memcpy(&v, &value, sizeof(T));
  s[lane] = v;
  warp_sync(mask);
  unsigned live = active_mask() & (mask | (1u << lane)), r = 0;
  for (unsigned i = 0; i < 32; ++i)
    if (((live >> i) & 1u) && s[i] == v) r |= 1u << i;
  warp_sync(mask);
  if (participants) *participants = live;
  return r;
}
}  // namespace cpu
}  // namespace pk
template <class T>
static inline unsigned __match_any_sync(unsigned mask, T value) { return ::pk::cpu::match(mask, value, nullptr); }
template <class T>
static inline unsigned __match_all_sync(unsigned mask, T value, int* pred) {
  unsigned live;
  unsigned m = ::pk::cpu::match(mask, value, &live);
  *pred = (m == live);
  return m == live ? live : 0u;
}

#define PK_WARP_REDUCE(NAME, T, OP)                                       \
  static inline T NAME(unsigned mask, T value) {                          \
    unsigned lane = ::pk::cpu::lane_id();                                 \
    uint64_t* s = ::pk::cpu::warp_slots();                                \
    s[lane] = (uint64_t)value;                                            \
    ::pk::cpu::warp_sync(mask);                                           \
    unsigned live = ::pk::cpu::active_mask() & mask;                      \
    bool first = true;                                                    \
    T r = value;                                                          \
    for (unsigned i = 0; i < 32; ++i)                                     \
      if ((live >> i) & 1u) {                                             \
        T v = (T)s[i];                                                    \
        if (first) { r = v; first = false; } else { r = OP; }             \
      }                                                                   \
    ::pk::cpu::warp_sync(mask);                                           \
    return r;                                                             \
  }
PK_WARP_REDUCE(__reduce_add_sync, unsigned, r + v)
PK_WARP_REDUCE(__reduce_add_sync, int, r + v)
PK_WARP_REDUCE(__reduce_min_sync, unsigned, (v < r ? v : r))
PK_WARP_REDUCE(__reduce_min_sync, int, (v < r ? v : r))
PK_WARP_REDUCE(__reduce_max_sync, unsigned, (v > r ? v : r))
PK_WARP_REDUCE(__reduce_max_sync, int, (v > r ? v : r))
PK_WARP_REDUCE(__reduce_and_sync, unsigned, (r & v))
PK_WARP_REDUCE(__reduce_or_sync, unsigned, (r | v))
PK_WARP_REDUCE(__reduce_xor_sync, unsigned, (r ^ v))
#undef PK_WARP_REDUCE

// ---------------------------------------------------------------------------
// Atomics (CUDA atomics are relaxed read-modify-write operations)
// ---------------------------------------------------------------------------
#define PK_RMW(NAME, T, BUILTIN) \
  static inline T NAME(T* p, T v) { return BUILTIN(p, v, __ATOMIC_RELAXED); }
PK_RMW(atomicAdd, int, __atomic_fetch_add)
PK_RMW(atomicAdd, unsigned int, __atomic_fetch_add)
PK_RMW(atomicAdd, unsigned long long, __atomic_fetch_add)
PK_RMW(atomicAdd, long long, __atomic_fetch_add)
PK_RMW(atomicSub, int, __atomic_fetch_sub)
PK_RMW(atomicSub, unsigned int, __atomic_fetch_sub)
PK_RMW(atomicSub, unsigned long long, __atomic_fetch_sub)
PK_RMW(atomicExch, int, __atomic_exchange_n)
PK_RMW(atomicExch, unsigned int, __atomic_exchange_n)
PK_RMW(atomicExch, unsigned long long, __atomic_exchange_n)
PK_RMW(atomicAnd, int, __atomic_fetch_and)
PK_RMW(atomicAnd, unsigned int, __atomic_fetch_and)
PK_RMW(atomicAnd, unsigned long long, __atomic_fetch_and)
PK_RMW(atomicOr, int, __atomic_fetch_or)
PK_RMW(atomicOr, unsigned int, __atomic_fetch_or)
PK_RMW(atomicOr, unsigned long long, __atomic_fetch_or)
PK_RMW(atomicXor, int, __atomic_fetch_xor)
PK_RMW(atomicXor, unsigned int, __atomic_fetch_xor)
PK_RMW(atomicXor, unsigned long long, __atomic_fetch_xor)
#undef PK_RMW

namespace pk {
template <class T, class F>
inline T atomic_update(T* p, F f) {
  T old;
  __atomic_load(p, &old, __ATOMIC_RELAXED);
  T nw;
  do {
    nw = f(old);
  } while (!__atomic_compare_exchange(p, &old, &nw, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
  return old;
}
}  // namespace pk

static inline float atomicAdd(float* p, float v) { return ::pk::atomic_update(p, [v](float o) { return o + v; }); }
static inline double atomicAdd(double* p, double v) { return ::pk::atomic_update(p, [v](double o) { return o + v; }); }
static inline float atomicExch(float* p, float v) {
  float old;
  __atomic_exchange(p, &v, &old, __ATOMIC_RELAXED);
  return old;
}
#define PK_MINMAX_ATOMIC(T)                                                                   \
  static inline T atomicMin(T* p, T v) { return ::pk::atomic_update(p, [v](T o) { return v < o ? v : o; }); } \
  static inline T atomicMax(T* p, T v) { return ::pk::atomic_update(p, [v](T o) { return v > o ? v : o; }); }
PK_MINMAX_ATOMIC(int)
PK_MINMAX_ATOMIC(unsigned int)
PK_MINMAX_ATOMIC(long long)
PK_MINMAX_ATOMIC(unsigned long long)
#undef PK_MINMAX_ATOMIC
#define PK_CAS(T)                                                                                      \
  static inline T atomicCAS(T* p, T compare, T val) {                                                  \
    __atomic_compare_exchange_n(p, &compare, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);         \
    return compare;                                                                                    \
  }
PK_CAS(int)
PK_CAS(unsigned int)
PK_CAS(unsigned long long)
PK_CAS(unsigned short)
#undef PK_CAS
static inline unsigned int atomicInc(unsigned int* p, unsigned int val) {
  return ::pk::atomic_update(p, [val](unsigned o) { return (o >= val) ? 0u : o + 1u; });
}
static inline unsigned int atomicDec(unsigned int* p, unsigned int val) {
  return ::pk::atomic_update(p, [val](unsigned o) { return (o == 0u || o > val) ? val : o - 1u; });
}
// Scoped variants behave like the device-wide ones on the CPU.
#define PK_SCOPED(NAME)                                                                        \
  template <class T, class... U>                                                               \
  static inline auto NAME##_block(T* p, U... v)->decltype(NAME(p, v...)) { return NAME(p, v...); } \
  template <class T, class... U>                                                               \
  static inline auto NAME##_system(T* p, U... v)->decltype(NAME(p, v...)) { return NAME(p, v...); }
PK_SCOPED(atomicAdd)
PK_SCOPED(atomicSub)
PK_SCOPED(atomicExch)
PK_SCOPED(atomicMin)
PK_SCOPED(atomicMax)
PK_SCOPED(atomicAnd)
PK_SCOPED(atomicOr)
PK_SCOPED(atomicXor)
PK_SCOPED(atomicCAS)
PK_SCOPED(atomicInc)
PK_SCOPED(atomicDec)
#undef PK_SCOPED

// ---------------------------------------------------------------------------
// min / max (CUDA provides these as global overloads in device code)
// ---------------------------------------------------------------------------
// Non-template overloads for every pair of arithmetic types (as in the CUDA headers):
// a template would stop kernels named min/max from being launched by name.
#define PK_MINMAX2(A, B)                                                                     \
  static inline std::common_type<A, B>::type min(A a, B b) {                               \
    typedef std::common_type<A, B>::type C;                                                 \
    return (C)a < (C)b ? (C)a : (C)b;                                                       \
  }                                                                                         \
  static inline std::common_type<A, B>::type max(A a, B b) {                               \
    typedef std::common_type<A, B>::type C;                                                 \
    return (C)a > (C)b ? (C)a : (C)b;                                                       \
  }
PK_MINMAX2(int, int) PK_MINMAX2(int, unsigned int) PK_MINMAX2(int, long) PK_MINMAX2(int, unsigned long) PK_MINMAX2(int, long long) PK_MINMAX2(int, unsigned long long) PK_MINMAX2(int, float) PK_MINMAX2(int, double)
PK_MINMAX2(unsigned int, int) PK_MINMAX2(unsigned int, unsigned int) PK_MINMAX2(unsigned int, long) PK_MINMAX2(unsigned int, unsigned long) PK_MINMAX2(unsigned int, long long) PK_MINMAX2(unsigned int, unsigned long long) PK_MINMAX2(unsigned int, float) PK_MINMAX2(unsigned int, double)
PK_MINMAX2(long, int) PK_MINMAX2(long, unsigned int) PK_MINMAX2(long, long) PK_MINMAX2(long, unsigned long) PK_MINMAX2(long, long long) PK_MINMAX2(long, unsigned long long) PK_MINMAX2(long, float) PK_MINMAX2(long, double)
PK_MINMAX2(unsigned long, int) PK_MINMAX2(unsigned long, unsigned int) PK_MINMAX2(unsigned long, long) PK_MINMAX2(unsigned long, unsigned long) PK_MINMAX2(unsigned long, long long) PK_MINMAX2(unsigned long, unsigned long long) PK_MINMAX2(unsigned long, float) PK_MINMAX2(unsigned long, double)
PK_MINMAX2(long long, int) PK_MINMAX2(long long, unsigned int) PK_MINMAX2(long long, long) PK_MINMAX2(long long, unsigned long) PK_MINMAX2(long long, long long) PK_MINMAX2(long long, unsigned long long) PK_MINMAX2(long long, float) PK_MINMAX2(long long, double)
PK_MINMAX2(unsigned long long, int) PK_MINMAX2(unsigned long long, unsigned int) PK_MINMAX2(unsigned long long, long) PK_MINMAX2(unsigned long long, unsigned long) PK_MINMAX2(unsigned long long, long long) PK_MINMAX2(unsigned long long, unsigned long long) PK_MINMAX2(unsigned long long, float) PK_MINMAX2(unsigned long long, double)
PK_MINMAX2(float, int) PK_MINMAX2(float, unsigned int) PK_MINMAX2(float, long) PK_MINMAX2(float, unsigned long) PK_MINMAX2(float, long long) PK_MINMAX2(float, unsigned long long) PK_MINMAX2(float, float) PK_MINMAX2(float, double)
PK_MINMAX2(double, int) PK_MINMAX2(double, unsigned int) PK_MINMAX2(double, long) PK_MINMAX2(double, unsigned long) PK_MINMAX2(double, long long) PK_MINMAX2(double, unsigned long long) PK_MINMAX2(double, float) PK_MINMAX2(double, double)
#undef PK_MINMAX2

// ---------------------------------------------------------------------------
// Math functions and intrinsics that are not part of the C library
// ---------------------------------------------------------------------------
static inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
static inline double rsqrt(double x) { return 1.0 / sqrt(x); }
static inline float rcbrtf(float x) { return 1.0f / cbrtf(x); }
static inline double rcbrt(double x) { return 1.0 / cbrt(x); }
static inline float sinpif(float x) { return sinf(3.14159265358979323846f * x); }
static inline float cospif(float x) { return cosf(3.14159265358979323846f * x); }
static inline double sinpi(double x) { return sin(3.14159265358979323846 * x); }
static inline double cospi(double x) { return cos(3.14159265358979323846 * x); }
static inline void sincospif(float x, float* s, float* c) { *s = sinpif(x); *c = cospif(x); }
static inline float norm3df(float a, float b, float c) { return sqrtf(a * a + b * b + c * c); }
static inline float rnorm3df(float a, float b, float c) { return 1.0f / sqrtf(a * a + b * b + c * c); }
static inline float norm4df(float a, float b, float c, float d) { return sqrtf(a * a + b * b + c * c + d * d); }
static inline double norm3d(double a, double b, double c) { return sqrt(a * a + b * b + c * c); }
static inline double rnorm3d(double a, double b, double c) { return 1.0 / sqrt(a * a + b * b + c * c); }
static inline float rhypotf(float a, float b) { return 1.0f / hypotf(a, b); }
static inline double rhypot(double a, double b) { return 1.0 / hypot(a, b); }
static inline float normcdff(float x) { return 0.5f * erfcf(-x * 0.70710678118654752f); }
static inline double normcdf(double x) { return 0.5 * erfc(-x * 0.70710678118654752); }
static inline float erfinvf(float x) {
  // M. Giles, "Approximating the erfinv function" (single precision)
  float w = -logf((1.0f - x) * (1.0f + x)), p;
  if (w < 5.0f) {
    w = w - 2.5f;
    p = 2.81022636e-08f; p = 3.43273939e-07f + p * w; p = -3.5233877e-06f + p * w;
    p = -4.39150654e-06f + p * w; p = 0.00021858087f + p * w; p = -0.00125372503f + p * w;
    p = -0.00417768164f + p * w; p = 0.246640727f + p * w; p = 1.50140941f + p * w;
  } else {
    w = sqrtf(w) - 3.0f;
    p = -0.000200214257f; p = 0.000100950558f + p * w; p = 0.00134934322f + p * w;
    p = -0.00367342844f + p * w; p = 0.00573950773f + p * w; p = -0.0076224613f + p * w;
    p = 0.00943887047f + p * w; p = 1.00167406f + p * w; p = 2.83297682f + p * w;
  }
  return p * x;
}
static inline double erfinv(double x) { return (double)erfinvf((float)x); }
static inline float erfcinvf(float x) { return erfinvf(1.0f - x); }
static inline float normcdfinvf(float p) { return -1.41421356237f * erfcinvf(2.0f * p); }
static inline float fdividef(float a, float b) { return a / b; }
static inline float __fdividef(float a, float b) { return a / b; }
// glibc declares these names internally, so they are provided as macros.
#define __expf(x) expf(x)
#define __exp10f(x) powf(10.0f, (x))
#define __logf(x) logf(x)
#define __log2f(x) log2f(x)
#define __log10f(x) log10f(x)
#define __sinf(x) sinf(x)
#define __cosf(x) cosf(x)
#define __tanf(x) tanf(x)
#define __sincosf(x, s, c) (*(s) = sinf(x), *(c) = cosf(x))
#define __powf(a, b) powf((a), (b))
static inline float __saturatef(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
static inline float __frcp_rn(float x) { return 1.0f / x; }
static inline float __fsqrt_rn(float x) { return sqrtf(x); }
static inline float __frsqrt_rn(float x) { return 1.0f / sqrtf(x); }
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __fmaf_rn(float a, float b, float c) { return fmaf(a, b, c); }
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline double __fma_rn(double a, double b, double c) { return fma(a, b, c); }
#if !defined(__GLIBC__)
static inline float exp10f(float x) { return powf(10.0f, x); }
static inline double exp10(double x) { return pow(10.0, x); }
#endif
#if !defined(__GLIBC__) && !defined(__MINGW32__)
static inline void sincosf(float x, float* s, float* c) { *s = sinf(x); *c = cosf(x); }
static inline void sincos(double x, double* s, double* c) { *s = sin(x); *c = cos(x); }
#endif

// Type reinterpretation and conversion intrinsics
namespace pk {
template <class To, class From>
inline To bitcast(From f) {
  static_assert(sizeof(To) == sizeof(From), "size mismatch");
  To t;
  std::memcpy(&t, &f, sizeof t);
  return t;
}
}  // namespace pk
static inline int __float_as_int(float x) { return ::pk::bitcast<int>(x); }
static inline unsigned __float_as_uint(float x) { return ::pk::bitcast<unsigned>(x); }
static inline float __int_as_float(int x) { return ::pk::bitcast<float>(x); }
static inline float __uint_as_float(unsigned x) { return ::pk::bitcast<float>(x); }
static inline long long __double_as_longlong(double x) { return ::pk::bitcast<long long>(x); }
static inline double __longlong_as_double(long long x) { return ::pk::bitcast<double>(x); }
static inline int __double2hiint(double x) { return (int)(::pk::bitcast<unsigned long long>(x) >> 32); }
static inline int __double2loint(double x) { return (int)(::pk::bitcast<unsigned long long>(x) & 0xffffffffu); }
static inline double __hiloint2double(int hi, int lo) {
  return ::pk::bitcast<double>(((unsigned long long)(unsigned)hi << 32) | (unsigned)lo);
}
static inline int __float2int_rn(float x) { return (int)nearbyintf(x); }
static inline int __float2int_rz(float x) { return (int)x; }
static inline int __float2int_rd(float x) { return (int)floorf(x); }
static inline int __float2int_ru(float x) { return (int)ceilf(x); }
static inline unsigned __float2uint_rn(float x) { return (unsigned)nearbyintf(x); }
static inline unsigned __float2uint_rz(float x) { return (unsigned)x; }
static inline long long __float2ll_rn(float x) { return (long long)nearbyintf(x); }
static inline long long __float2ll_rz(float x) { return (long long)x; }
static inline float __int2float_rn(int x) { return (float)x; }
static inline float __uint2float_rn(unsigned x) { return (float)x; }
static inline float __ll2float_rn(long long x) { return (float)x; }
static inline float __double2float_rn(double x) { return (float)x; }
static inline int __double2int_rn(double x) { return (int)nearbyint(x); }
static inline int __double2int_rz(double x) { return (int)x; }
static inline double __int2double_rn(int x) { return (double)x; }

// Integer intrinsics
static inline int __popc(unsigned x) { return __builtin_popcount(x); }
static inline int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
static inline int __clz(int x) { return x == 0 ? 32 : __builtin_clz((unsigned)x); }
static inline int __clzll(long long x) { return x == 0 ? 64 : __builtin_clzll((unsigned long long)x); }
static inline int __ffs(int x) { return __builtin_ffs(x); }
static inline int __ffsll(long long x) { return __builtin_ffsll(x); }
static inline unsigned __brev(unsigned x) {
  x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
  x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
  x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
  x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
  return (x >> 16) | (x << 16);
}
static inline unsigned long long __brevll(unsigned long long x) {
  return ((unsigned long long)__brev((unsigned)x) << 32) | __brev((unsigned)(x >> 32));
}
static inline unsigned __byte_perm(unsigned x, unsigned y, unsigned s) {
  unsigned long long v = ((unsigned long long)y << 32) | x;
  unsigned r = 0;
  for (int i = 0; i < 4; ++i) r |= (unsigned)((v >> (8 * ((s >> (4 * i)) & 7u))) & 0xffu) << (8 * i);
  return r;
}
static inline int __mul24(int a, int b) { return ((a << 8) >> 8) * ((b << 8) >> 8); }
static inline unsigned __umul24(unsigned a, unsigned b) { return (a & 0xffffffu) * (b & 0xffffffu); }
static inline int __mulhi(int a, int b) { return (int)(((long long)a * b) >> 32); }
static inline unsigned __umulhi(unsigned a, unsigned b) { return (unsigned)(((unsigned long long)a * b) >> 32); }
static inline unsigned long long __umul64hi(unsigned long long a, unsigned long long b) {
  unsigned long long a_lo = (unsigned)a, a_hi = a >> 32, b_lo = (unsigned)b, b_hi = b >> 32;
  unsigned long long p0 = a_lo * b_lo, p1 = a_lo * b_hi, p2 = a_hi * b_lo, p3 = a_hi * b_hi;
  unsigned long long mid = (p0 >> 32) + (unsigned)p1 + (unsigned)p2;
  return p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
}
static inline long long __mul64hi(long long a, long long b) {
  unsigned long long r = __umul64hi((unsigned long long)a, (unsigned long long)b);
  if (a < 0) r -= (unsigned long long)b;
  if (b < 0) r -= (unsigned long long)a;
  return (long long)r;
}
static inline int __hadd(int a, int b) { return (int)(((long long)a + b) >> 1); }
static inline int __rhadd(int a, int b) { return (int)(((long long)a + b + 1) >> 1); }
static inline unsigned __uhadd(unsigned a, unsigned b) { return (unsigned)(((unsigned long long)a + b) >> 1); }
static inline unsigned __urhadd(unsigned a, unsigned b) { return (unsigned)(((unsigned long long)a + b + 1) >> 1); }
static inline unsigned __sad(int x, int y, unsigned z) { return (unsigned)(x > y ? x - y : y - x) + z; }
static inline unsigned __usad(unsigned x, unsigned y, unsigned z) { return (x > y ? x - y : y - x) + z; }
static inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned s) {
  unsigned long long v = ((unsigned long long)hi << 32) | lo;
  return (unsigned)((v << (s & 31u)) >> 32);
}
static inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned s) {
  unsigned long long v = ((unsigned long long)hi << 32) | lo;
  return (unsigned)(v >> (s & 31u));
}

// Memory access hints: plain loads/stores on the CPU
template <class T> static inline T __ldg(const T* p) { return *p; }
template <class T> static inline T __ldcg(const T* p) { return *p; }
template <class T> static inline T __ldca(const T* p) { return *p; }
template <class T> static inline T __ldcs(const T* p) { return *p; }
template <class T> static inline T __ldlu(const T* p) { return *p; }
template <class T> static inline T __ldcv(const T* p) { return *p; }
template <class T> static inline void __stwb(T* p, T v) { *p = v; }
template <class T> static inline void __stcg(T* p, T v) { *p = v; }
template <class T> static inline void __stcs(T* p, T v) { *p = v; }
template <class T> static inline void __stwt(T* p, T v) { *p = v; }

// Timing / misc
static inline long long clock64() {
  return (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
static inline void __nanosleep(unsigned) {}
[[noreturn]] static inline void __trap() { abort(); }
static inline void __brkpt() {}
static inline void __prof_trigger(int) {}

#endif  // POLYKERNEL_PK_DEVICE_H
