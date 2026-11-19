/* ---- PolyKernel OpenCL C prelude: CUDA device built-ins on OpenCL ---- */
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define PK_HAS_FP64 1
#endif
#if defined(cl_khr_int64_base_atomics)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#define PK_HAS_I64_ATOMICS 1
#endif
#if defined(cl_khr_int64_extended_atomics)
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#define PK_HAS_I64_EXT_ATOMICS 1
#endif
#if (defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ == 200) || defined(__opencl_c_generic_address_space)
#define PK_GENERIC 1
#define PK_AS
#else
#define PK_AS __global
#endif
#ifndef NULL
#define NULL ((void*)0)
#endif

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;
typedef long int64_t;
typedef ulong uint64_t;
typedef long ssize_t;

/* CUDA's 1- and 3-component vectors are plain structs (OpenCL's float3 is 16 bytes) */
#define PK_VEC1(T, N) typedef struct { T x; } pk_##N##1; \
  static inline pk_##N##1 pk_make_##N##1(T x) { pk_##N##1 r; r.x = x; return r; }
#define PK_VEC3(T, N) typedef struct { T x, y, z; } pk_##N##3; \
  static inline pk_##N##3 pk_make_##N##3(T x, T y, T z) { pk_##N##3 r; r.x = x; r.y = y; r.z = z; return r; }
PK_VEC1(char, char) PK_VEC1(uchar, uchar) PK_VEC1(short, short) PK_VEC1(ushort, ushort)
PK_VEC1(int, int) PK_VEC1(uint, uint) PK_VEC1(long, long) PK_VEC1(ulong, ulong) PK_VEC1(float, float)
PK_VEC3(char, char) PK_VEC3(uchar, uchar) PK_VEC3(short, short) PK_VEC3(ushort, ushort)
PK_VEC3(int, int) PK_VEC3(uint, uint) PK_VEC3(long, long) PK_VEC3(ulong, ulong) PK_VEC3(float, float)
#ifdef PK_HAS_FP64
PK_VEC1(double, double) PK_VEC3(double, double)
#endif
/* 2- and 4-component vectors map to OpenCL vectors (same size, alignment and .x/.y/.z/.w) */
#define make_char2(a, b) ((char2)((a), (b)))
#define make_uchar2(a, b) ((uchar2)((a), (b)))
#define make_short2(a, b) ((short2)((a), (b)))
#define make_ushort2(a, b) ((ushort2)((a), (b)))
#define make_int2(a, b) ((int2)((a), (b)))
#define make_uint2(a, b) ((uint2)((a), (b)))
#define make_long2(a, b) ((long2)((a), (b)))
#define make_ulong2(a, b) ((ulong2)((a), (b)))
#define make_longlong2(a, b) ((long2)((a), (b)))
#define make_ulonglong2(a, b) ((ulong2)((a), (b)))
#define make_float2(a, b) ((float2)((a), (b)))
#define make_double2(a, b) ((double2)((a), (b)))
#define make_char4(a, b, c, d) ((char4)((a), (b), (c), (d)))
#define make_uchar4(a, b, c, d) ((uchar4)((a), (b), (c), (d)))
#define make_short4(a, b, c, d) ((short4)((a), (b), (c), (d)))
#define make_ushort4(a, b, c, d) ((ushort4)((a), (b), (c), (d)))
#define make_int4(a, b, c, d) ((int4)((a), (b), (c), (d)))
#define make_uint4(a, b, c, d) ((uint4)((a), (b), (c), (d)))
#define make_long4(a, b, c, d) ((long4)((a), (b), (c), (d)))
#define make_ulong4(a, b, c, d) ((ulong4)((a), (b), (c), (d)))
#define make_longlong4(a, b, c, d) ((long4)((a), (b), (c), (d)))
#define make_ulonglong4(a, b, c, d) ((ulong4)((a), (b), (c), (d)))
#define make_float4(a, b, c, d) ((float4)((a), (b), (c), (d)))
#define make_double4(a, b, c, d) ((double4)((a), (b), (c), (d)))

/* ---- math: CUDA's f-suffixed names (arguments converted to float like the C prototypes) ---- */
#define PK_F(x) ((float)(x))
#define sqrtf(x) sqrt(PK_F(x))
#define rsqrtf(x) rsqrt(PK_F(x))
#define cbrtf(x) cbrt(PK_F(x))
#define rcbrtf(x) (1.0f / cbrt(PK_F(x)))
#define expf(x) exp(PK_F(x))
#define exp2f(x) exp2(PK_F(x))
#define exp10f(x) exp10(PK_F(x))
#define expm1f(x) expm1(PK_F(x))
#define logf(x) log(PK_F(x))
#define log2f(x) log2(PK_F(x))
#define log10f(x) log10(PK_F(x))
#define log1pf(x) log1p(PK_F(x))
#define logbf(x) logb(PK_F(x))
#define powf(x, y) pow(PK_F(x), PK_F(y))
#define sinf(x) sin(PK_F(x))
#define cosf(x) cos(PK_F(x))
#define tanf(x) tan(PK_F(x))
#define asinf(x) asin(PK_F(x))
#define acosf(x) acos(PK_F(x))
#define atanf(x) atan(PK_F(x))
#define atan2f(y, x) atan2(PK_F(y), PK_F(x))
#define sinhf(x) sinh(PK_F(x))
#define coshf(x) cosh(PK_F(x))
#define tanhf(x) tanh(PK_F(x))
#define asinhf(x) asinh(PK_F(x))
#define acoshf(x) acosh(PK_F(x))
#define atanhf(x) atanh(PK_F(x))
#define sinpif(x) sinpi(PK_F(x))
#define cospif(x) cospi(PK_F(x))
#define fabsf(x) fabs(PK_F(x))
#define floorf(x) floor(PK_F(x))
#define ceilf(x) ceil(PK_F(x))
#define truncf(x) trunc(PK_F(x))
#define roundf(x) round(PK_F(x))
#define rintf(x) rint(PK_F(x))
#define nearbyintf(x) rint(PK_F(x))
#define fmodf(x, y) fmod(PK_F(x), PK_F(y))
#define remainderf(x, y) remainder(PK_F(x), PK_F(y))
#define fminf(x, y) fmin(PK_F(x), PK_F(y))
#define fmaxf(x, y) fmax(PK_F(x), PK_F(y))
#define fdimf(x, y) fdim(PK_F(x), PK_F(y))
#define fmaf(x, y, z) fma(PK_F(x), PK_F(y), PK_F(z))
#define hypotf(x, y) hypot(PK_F(x), PK_F(y))
#define copysignf(x, y) copysign(PK_F(x), PK_F(y))
#define erff(x) erf(PK_F(x))
#define erfcf(x) erfc(PK_F(x))
#define lgammaf(x) lgamma(PK_F(x))
#define tgammaf(x) tgamma(PK_F(x))
#define ldexpf(x, e) ldexp(PK_F(x), (int)(e))
#define sincosf(x, s, c) (*(s) = sin(PK_F(x)), *(c) = cos(PK_F(x)))
#define sincospif(x, s, c) (*(s) = sinpi(PK_F(x)), *(c) = cospi(PK_F(x)))
#define norm3df(a, b, c) sqrt(PK_F(a) * PK_F(a) + PK_F(b) * PK_F(b) + PK_F(c) * PK_F(c))
#define rnorm3df(a, b, c) rsqrt(PK_F(a) * PK_F(a) + PK_F(b) * PK_F(b) + PK_F(c) * PK_F(c))
#define rhypotf(a, b) rsqrt(PK_F(a) * PK_F(a) + PK_F(b) * PK_F(b))
#define normcdff(x) (0.5f * erfc(-PK_F(x) * 0.70710678118654752f))
#define fdividef(a, b) (PK_F(a) / PK_F(b))
#define __fdividef(a, b) native_divide(PK_F(a), PK_F(b))
#define __expf(x) native_exp(PK_F(x))
#define __exp10f(x) native_exp10(PK_F(x))
#define __logf(x) native_log(PK_F(x))
#define __log2f(x) native_log2(PK_F(x))
#define __log10f(x) native_log10(PK_F(x))
#define __sinf(x) native_sin(PK_F(x))
#define __cosf(x) native_cos(PK_F(x))
#define __tanf(x) native_tan(PK_F(x))
#define __powf(x, y) native_powr(PK_F(x), PK_F(y))
#define __sincosf(x, s, c) (*(s) = native_sin(PK_F(x)), *(c) = native_cos(PK_F(x)))
#define __saturatef(x) clamp(PK_F(x), 0.0f, 1.0f)
#define __frcp_rn(x) (1.0f / PK_F(x))
#define __fsqrt_rn(x) sqrt(PK_F(x))
#define __frsqrt_rn(x) rsqrt(PK_F(x))
#define __fadd_rn(a, b) (PK_F(a) + PK_F(b))
#define __fsub_rn(a, b) (PK_F(a) - PK_F(b))
#define __fmul_rn(a, b) (PK_F(a) * PK_F(b))
#define __fdiv_rn(a, b) (PK_F(a) / PK_F(b))
#define __fmaf_rn(a, b, c) fma(PK_F(a), PK_F(b), PK_F(c))
static inline float __attribute__((overloadable)) erfinvf(float x) {
  float w = -log((1.0f - x) * (1.0f + x)), p;
  if (w < 5.0f) {
    w = w - 2.5f;
    p = 2.81022636e-08f; p = 3.43273939e-07f + p * w; p = -3.5233877e-06f + p * w;
    p = -4.39150654e-06f + p * w; p = 0.00021858087f + p * w; p = -0.00125372503f + p * w;
    p = -0.00417768164f + p * w; p = 0.246640727f + p * w; p = 1.50140941f + p * w;
  } else {
    w = sqrt(w) - 3.0f;
    p = -0.000200214257f; p = 0.000100950558f + p * w; p = 0.00134934322f + p * w;
    p = -0.00367342844f + p * w; p = 0.00573950773f + p * w; p = -0.0076224613f + p * w;
    p = 0.00943887047f + p * w; p = 1.00167406f + p * w; p = 2.83297682f + p * w;
  }
  return p * x;
}
#define erfcinvf(x) erfinvf(1.0f - PK_F(x))
#define normcdfinvf(p) (-1.41421356237f * erfcinvf(2.0f * PK_F(p)))
#ifdef PK_HAS_FP64
#define rcbrt(x) (1.0 / cbrt((double)(x)))
#define norm3d(a, b, c) sqrt((double)(a) * (a) + (double)(b) * (b) + (double)(c) * (c))
#define rnorm3d(a, b, c) rsqrt((double)(a) * (a) + (double)(b) * (b) + (double)(c) * (c))
#define normcdf(x) (0.5 * erfc(-(double)(x) * 0.70710678118654752))
#endif

/* ---- cooperative_groups::thread_block (block-level groups only) ---- */
typedef struct { int pk_unused; } pk_thread_block;
static inline pk_thread_block pk_this_thread_block(void) { pk_thread_block b; b.pk_unused = 0; return b; }
#define pk_block_rank() ((uint)(get_local_id(0) + get_local_size(0) * (get_local_id(1) + get_local_size(1) * get_local_id(2))))
#define pk_block_size() ((uint)(get_local_size(0) * get_local_size(1) * get_local_size(2)))

/* ---- conversion and bit intrinsics ---- */
#define __float_as_int(x) as_int(PK_F(x))
#define __float_as_uint(x) as_uint(PK_F(x))
#define __int_as_float(x) as_float((int)(x))
#define __uint_as_float(x) as_float((uint)(x))
#define __float2int_rn(x) convert_int_rte(PK_F(x))
#define __float2int_rz(x) convert_int_rtz(PK_F(x))
#define __float2int_rd(x) convert_int_rtn(PK_F(x))
#define __float2int_ru(x) convert_int_rtp(PK_F(x))
#define __float2uint_rn(x) convert_uint_rte(PK_F(x))
#define __float2uint_rz(x) convert_uint_rtz(PK_F(x))
#define __float2ll_rn(x) convert_long_rte(PK_F(x))
#define __float2ll_rz(x) convert_long_rtz(PK_F(x))
#define __int2float_rn(x) convert_float_rte((int)(x))
#define __uint2float_rn(x) convert_float_rte((uint)(x))
#define __ll2float_rn(x) convert_float_rte((long)(x))
#ifdef PK_HAS_FP64
#define __double_as_longlong(x) as_long((double)(x))
#define __longlong_as_double(x) as_double((long)(x))
#define __double2float_rn(x) convert_float_rte((double)(x))
#define __double2int_rn(x) convert_int_rte((double)(x))
#define __double2int_rz(x) convert_int_rtz((double)(x))
#define __int2double_rn(x) ((double)(int)(x))
#define __double2hiint(x) ((int)(as_ulong((double)(x)) >> 32))
#define __double2loint(x) ((int)(as_ulong((double)(x)) & 0xffffffffu))
#define __hiloint2double(h, l) as_double(((ulong)(uint)(h) << 32) | (ulong)(uint)(l))
#endif
#define __popc(x) ((int)popcount((uint)(x)))
#define __popcll(x) ((int)popcount((ulong)(x)))
#define __clz(x) ((int)clz((int)(x)))
#define __clzll(x) ((int)clz((long)(x)))
#define __ffs(x) ((int)((uint)(x) == 0u ? 0 : 32 - clz((uint)(x) & (0u - (uint)(x)))))
#define __ffsll(x) ((int)((ulong)(x) == 0ul ? 0 : 64 - clz((ulong)(x) & (0ul - (ulong)(x)))))
static inline uint pk_brev(uint x) {
  x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
  x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
  x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
  x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
  return (x >> 16) | (x << 16);
}
#define __brev(x) pk_brev((uint)(x))
#define __brevll(x) (((ulong)pk_brev((uint)(x)) << 32) | (ulong)pk_brev((uint)((ulong)(x) >> 32)))
static inline uint pk_byte_perm(uint x, uint y, uint s) {
  ulong v = ((ulong)y << 32) | x;
  uint r = 0;
  for (int i = 0; i < 4; ++i) r |= (uint)((v >> (8 * ((s >> (4 * i)) & 7u))) & 0xffu) << (8 * i);
  return r;
}
#define __byte_perm(x, y, s) pk_byte_perm((uint)(x), (uint)(y), (uint)(s))
#define __mul24(a, b) mul24((int)(a), (int)(b))
#define __umul24(a, b) mul24((uint)(a), (uint)(b))
#define __mulhi(a, b) mul_hi((int)(a), (int)(b))
#define __umulhi(a, b) mul_hi((uint)(a), (uint)(b))
#define __mul64hi(a, b) mul_hi((long)(a), (long)(b))
#define __umul64hi(a, b) mul_hi((ulong)(a), (ulong)(b))
#define __hadd(a, b) hadd((int)(a), (int)(b))
#define __rhadd(a, b) rhadd((int)(a), (int)(b))
#define __uhadd(a, b) hadd((uint)(a), (uint)(b))
#define __urhadd(a, b) rhadd((uint)(a), (uint)(b))
#define __sad(x, y, z) (abs_diff((int)(x), (int)(y)) + (uint)(z))
#define __usad(x, y, z) (abs_diff((uint)(x), (uint)(y)) + (uint)(z))
#define __funnelshift_l(lo, hi, s) ((uint)(((((ulong)(uint)(hi)) << 32 | (uint)(lo)) << ((uint)(s) & 31u)) >> 32))
#define __funnelshift_r(lo, hi, s) ((uint)((((ulong)(uint)(hi)) << 32 | (uint)(lo)) >> ((uint)(s) & 31u)))
#define __ldg(p) (*(p))
#define __ldcg(p) (*(p))
#define __ldca(p) (*(p))
#define __ldcs(p) (*(p))
#define __ldlu(p) (*(p))
#define __ldcv(p) (*(p))
#define __threadfence() mem_fence(CLK_GLOBAL_MEM_FENCE)
#define __threadfence_block() mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
#define __threadfence_system() mem_fence(CLK_GLOBAL_MEM_FENCE)
#define clock() ((long)0)
#define clock64() ((long)0)
#define __trap() ((void)0)
#define __brkpt() ((void)0)
#define __prof_trigger(x) ((void)0)
#define __nanosleep(x) ((void)0)

/* ---- atomics (CUDA atomics return the old value) ---- */
#define PK_ATOM_AS(AS)                                                                                         \
  static inline int __attribute__((overloadable)) atomicAdd(volatile AS int* p, int v) { return atomic_add(p, v); } \
  static inline uint __attribute__((overloadable)) atomicAdd(volatile AS uint* p, uint v) { return atomic_add(p, v); } \
  static inline int __attribute__((overloadable)) atomicSub(volatile AS int* p, int v) { return atomic_sub(p, v); } \
  static inline uint __attribute__((overloadable)) atomicSub(volatile AS uint* p, uint v) { return atomic_sub(p, v); } \
  static inline int __attribute__((overloadable)) atomicExch(volatile AS int* p, int v) { return atomic_xchg(p, v); } \
  static inline uint __attribute__((overloadable)) atomicExch(volatile AS uint* p, uint v) { return atomic_xchg(p, v); } \
  static inline float __attribute__((overloadable)) atomicExch(volatile AS float* p, float v) { return atomic_xchg(p, v); } \
  static inline int __attribute__((overloadable)) atomicMin(volatile AS int* p, int v) { return atomic_min(p, v); } \
  static inline uint __attribute__((overloadable)) atomicMin(volatile AS uint* p, uint v) { return atomic_min(p, v); } \
  static inline int __attribute__((overloadable)) atomicMax(volatile AS int* p, int v) { return atomic_max(p, v); } \
  static inline uint __attribute__((overloadable)) atomicMax(volatile AS uint* p, uint v) { return atomic_max(p, v); } \
  static inline int __attribute__((overloadable)) atomicAnd(volatile AS int* p, int v) { return atomic_and(p, v); } \
  static inline uint __attribute__((overloadable)) atomicAnd(volatile AS uint* p, uint v) { return atomic_and(p, v); } \
  static inline int __attribute__((overloadable)) atomicOr(volatile AS int* p, int v) { return atomic_or(p, v); } \
  static inline uint __attribute__((overloadable)) atomicOr(volatile AS uint* p, uint v) { return atomic_or(p, v); } \
  static inline int __attribute__((overloadable)) atomicXor(volatile AS int* p, int v) { return atomic_xor(p, v); } \
  static inline uint __attribute__((overloadable)) atomicXor(volatile AS uint* p, uint v) { return atomic_xor(p, v); } \
  static inline int __attribute__((overloadable)) atomicCAS(volatile AS int* p, int c, int v) { return atomic_cmpxchg(p, c, v); } \
  static inline uint __attribute__((overloadable)) atomicCAS(volatile AS uint* p, uint c, uint v) { return atomic_cmpxchg(p, c, v); } \
  static inline float __attribute__((overloadable)) atomicAdd(volatile AS float* p, float v) {               \
    uint o, n;                                                                                               \
    do { o = as_uint(*p); n = as_uint(as_float(o) + v); } while (atomic_cmpxchg((volatile AS uint*)p, o, n) != o); \
    return as_float(o);                                                                                      \
  }                                                                                                          \
  static inline uint __attribute__((overloadable)) atomicInc(volatile AS uint* p, uint lim) {                \
    uint o, n;                                                                                               \
    do { o = *p; n = (o >= lim) ? 0u : o + 1u; } while (atomic_cmpxchg(p, o, n) != o);                      \
    return o;                                                                                                \
  }                                                                                                          \
  static inline uint __attribute__((overloadable)) atomicDec(volatile AS uint* p, uint lim) {                \
    uint o, n;                                                                                               \
    do { o = *p; n = (o == 0u || o > lim) ? lim : o - 1u; } while (atomic_cmpxchg(p, o, n) != o);           \
    return o;                                                                                                \
  }
PK_ATOM_AS(__global)
PK_ATOM_AS(__local)

#if defined(PK_HAS_I64_ATOMICS)
#define PK_ATOM64_AS(AS)                                                                                         \
  static inline ulong __attribute__((overloadable)) atomicAdd(volatile AS ulong* p, ulong v) { return atom_add(p, v); } \
  static inline long __attribute__((overloadable)) atomicAdd(volatile AS long* p, long v) { return atom_add(p, v); } \
  static inline ulong __attribute__((overloadable)) atomicExch(volatile AS ulong* p, ulong v) { return atom_xchg(p, v); } \
  static inline ulong __attribute__((overloadable)) atomicCAS(volatile AS ulong* p, ulong c, ulong v) { return atom_cmpxchg(p, c, v); }
PK_ATOM64_AS(__global)
#if defined(PK_HAS_FP64)
static inline double __attribute__((overloadable)) atomicAdd(volatile __global double* p, double v) {
  ulong o, n;
  do { o = as_ulong(*p); n = as_ulong(as_double(o) + v); } while (atom_cmpxchg((volatile __global ulong*)p, o, n) != o);
  return as_double(o);
}
#endif
#endif

#if !defined(PK_GENERIC)
/* OpenCL C 1.2: device functions are emitted once per address space, so the private-pointer copy
   needs atomics too. Private memory belongs to one work-item, so a plain update is exact. */
#define PK_PRMW(NAME, T, EXPR)                                                                         \
  static inline T __attribute__((overloadable)) NAME(volatile __private T* p, T v) { T o = *p; *p = (EXPR); return o; }
#define PK_PRMW_INT(T)                                                                                 \
  PK_PRMW(atomicAdd, T, o + v) PK_PRMW(atomicSub, T, o - v) PK_PRMW(atomicExch, T, v)                \
  PK_PRMW(atomicMin, T, v < o ? v : o) PK_PRMW(atomicMax, T, v > o ? v : o) PK_PRMW(atomicAnd, T, o & v) \
  PK_PRMW(atomicOr, T, o | v) PK_PRMW(atomicXor, T, o ^ v)                                           \
  static inline T __attribute__((overloadable)) atomicCAS(volatile __private T* p, T c, T v) { T o = *p; if (o == c) *p = v; return o; }
PK_PRMW_INT(int)
PK_PRMW_INT(uint)
PK_PRMW_INT(long)
PK_PRMW_INT(ulong)
PK_PRMW(atomicAdd, float, o + v)
PK_PRMW(atomicExch, float, v)
#if defined(PK_HAS_FP64)
PK_PRMW(atomicAdd, double, o + v)
#endif
static inline uint __attribute__((overloadable)) atomicInc(volatile __private uint* p, uint lim) { uint o = *p; *p = (o >= lim) ? 0u : o + 1u; return o; }
static inline uint __attribute__((overloadable)) atomicDec(volatile __private uint* p, uint lim) { uint o = *p; *p = (o == 0u || o > lim) ? lim : o - 1u; return o; }
#endif

#if defined(PK_GENERIC) && (__OPENCL_C_VERSION__ == 200 || defined(__opencl_c_atomic_scope_device))
/* pointers of unknown address space (e.g. device-function parameters) */
#define PK_GRMW(NAME, T, AT, OP)                                                                       \
  static inline T __attribute__((overloadable)) NAME(volatile __generic T* p, T v) {                   \
    return atomic_##OP##_explicit((volatile __generic AT*)p, v, memory_order_relaxed, memory_scope_device); \
  }
PK_GRMW(atomicAdd, int, atomic_int, fetch_add)
PK_GRMW(atomicAdd, uint, atomic_uint, fetch_add)
PK_GRMW(atomicSub, int, atomic_int, fetch_sub)
PK_GRMW(atomicSub, uint, atomic_uint, fetch_sub)
PK_GRMW(atomicExch, int, atomic_int, exchange)
PK_GRMW(atomicExch, uint, atomic_uint, exchange)
PK_GRMW(atomicMin, int, atomic_int, fetch_min)
PK_GRMW(atomicMin, uint, atomic_uint, fetch_min)
PK_GRMW(atomicMax, int, atomic_int, fetch_max)
PK_GRMW(atomicMax, uint, atomic_uint, fetch_max)
PK_GRMW(atomicAnd, int, atomic_int, fetch_and)
PK_GRMW(atomicAnd, uint, atomic_uint, fetch_and)
PK_GRMW(atomicOr, int, atomic_int, fetch_or)
PK_GRMW(atomicOr, uint, atomic_uint, fetch_or)
PK_GRMW(atomicXor, int, atomic_int, fetch_xor)
PK_GRMW(atomicXor, uint, atomic_uint, fetch_xor)
static inline uint __attribute__((overloadable)) pk_gcas(volatile __generic uint* p, uint c, uint v) {
  atomic_compare_exchange_strong_explicit((volatile __generic atomic_uint*)p, &c, v, memory_order_relaxed,
                                          memory_order_relaxed, memory_scope_device);
  return c;
}
static inline int __attribute__((overloadable)) atomicCAS(volatile __generic int* p, int c, int v) {
  return (int)pk_gcas((volatile __generic uint*)p, (uint)c, (uint)v);
}
static inline uint __attribute__((overloadable)) atomicCAS(volatile __generic uint* p, uint c, uint v) {
  return pk_gcas(p, c, v);
}
static inline float __attribute__((overloadable)) atomicAdd(volatile __generic float* p, float v) {
  uint o, n;
  do { o = as_uint(*p); n = as_uint(as_float(o) + v); } while (pk_gcas((volatile __generic uint*)p, o, n) != o);
  return as_float(o);
}
static inline float __attribute__((overloadable)) atomicExch(volatile __generic float* p, float v) {
  return as_float(atomicExch((volatile __generic uint*)p, as_uint(v)));
}
#endif
/* ---- end of prelude ---- */
