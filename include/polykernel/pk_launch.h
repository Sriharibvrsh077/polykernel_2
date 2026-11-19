// PolyKernel: kernel launch machinery.
// pkcc rewrites   kernel<<<grid, block, shmem, stream>>>(args...)
// into            ::pk::launch_k(kernel, "kernel", &::pk_tu_info, ::pk::LaunchCfg(grid, block, shmem, stream), args...)
// (or ::pk::launch_generic for template kernels / function pointers).
#ifndef POLYKERNEL_PK_LAUNCH_H
#define POLYKERNEL_PK_LAUNCH_H

#include "pk_types.h"

namespace pk {

struct LaunchCfg {
  dim3 grid, block;
  size_t shmem;
  cudaStream_t stream;
  LaunchCfg(dim3 g, dim3 b, size_t s = 0, cudaStream_t st = 0) : grid(g), block(b), shmem(s), stream(st) {}
};

// Metadata emitted by the translator for each translation unit.
struct KernelMeta {     // a kernel that has an OpenCL C translation
  const char* name;
  const char* sig;      // comma separated: P (pointer) | I<n> (integer) | F4 | F8 | S (raw struct)
  const char* syms;     // comma separated __device__/__constant__ variables passed as hidden buffer arguments
};
struct SymbolMeta {     // a __device__ / __constant__ variable
  const void* host_addr;
  size_t size;
  const char* name;
};
struct TuInfo;
struct LaunchCfg;
typedef cudaError_t (*VoidArgsLauncher)(const void* fn, const char* name, const TuInfo* tu, const LaunchCfg& cfg,
                                        void** args);
struct KernelEntry {    // host function address -> launcher (for cudaLaunchKernel)
  const void* fn;
  const char* name;
  VoidArgsLauncher launcher;
};

struct TuInfo {
  const char* file = nullptr;
  bool needs_fibers = true;          // kernels in this TU use __syncthreads / warp intrinsics
  bool cl_ok = false;                // OpenCL C translation available
  const char* cl_reason = nullptr;   // why not (for diagnostics)
  const char* cl_src = nullptr;
  const KernelMeta* kernels = nullptr;
  int nkernels = 0;
  const char* const* launched = nullptr;   // kernel names launched from this TU
  int nlaunched = 0;
  bool has_generic_launch = false;   // a launch the OpenCL path cannot serve (deduced template kernel)
  bool has_dyn_launch = false;       // launches through function pointers (resolved by address at run time)
  bool has_template_kernels = false;
  const SymbolMeta* symbols = nullptr;
  int nsymbols = 0;
  const KernelEntry* entries = nullptr;
  int nentries = 0;
  bool ptx_ok = false;               // PTX module from PolyKernel's own compiler (NVIDIA driver back end)
  const char* ptx_reason = nullptr;
  const char* ptx_src = nullptr;
  const char* const* ptx_kernels = nullptr;
  int nptx_kernels = 0;
  bool registered = false;
  void* backend_data = nullptr;
  TuInfo* next = nullptr;
};

void register_tu(TuInfo* tu);
bool opencl_active();   // true when a GPU back end (PTX or OpenCL) runs kernels; decides on first call
const char* kernel_name_for(const void* host_fn);   // kernel name registered for a host function address

// Kernel arguments marshalled for the OpenCL backend.
struct ArgPack {
  enum Kind : unsigned char { Ptr, Int, Flt, Dbl, Raw };
  struct Arg {
    Kind kind;
    bool is_signed;
    unsigned size;
    uint64_t bits;
    const void* data;
  };
  enum { kMax = 64 };
  Arg args[kMax];
  int n = 0;
  bool overflow = false;

  template <class T>
  void add(const T& v) {
    if (n >= kMax) { overflow = true; return; }
    Arg& a = args[n++];
    a.is_signed = false;
    a.bits = 0;
    a.data = nullptr;
    a.size = (unsigned)sizeof(T);
    if constexpr (std::is_pointer<T>::value) {
      a.kind = Ptr;
      a.data = (const void*)v;
    } else if constexpr (std::is_same<T, float>::value) {
      a.kind = Flt;
      double d = v;
      std::memcpy(&a.bits, &d, sizeof d);
    } else if constexpr (std::is_floating_point<T>::value) {
      a.kind = Dbl;
      double d = (double)v;
      std::memcpy(&a.bits, &d, sizeof d);
    } else if constexpr (std::is_enum<T>::value) {
      typedef typename std::underlying_type<T>::type U;
      a.kind = Int;
      a.is_signed = std::is_signed<U>::value;
      a.bits = (uint64_t)(int64_t)(U)v;
      if (!a.is_signed) a.bits = (uint64_t)(U)v;
    } else if constexpr (std::is_integral<T>::value) {
      a.kind = Int;
      a.is_signed = std::is_signed<T>::value;
      a.bits = a.is_signed ? (uint64_t)(int64_t)v : (uint64_t)v;
    } else {
      a.kind = Raw;
      a.data = (const void*)std::addressof(v);
    }
  }
};

cudaError_t cpu_launch(const TuInfo* tu, const LaunchCfg& cfg, void (*thunk)(void*), void* ctx);
cudaError_t opencl_launch(const TuInfo* tu, const char* name, const LaunchCfg& cfg, const ArgPack& args);
cudaError_t gpu_launch(const TuInfo* tu, const char* name, const LaunchCfg& cfg, const ArgPack& args);
cudaError_t record_launch_error(cudaError_t e);

// Implicit conversion of a launch argument to the kernel parameter type
// (like a function call), with NULL/0 accepted for pointer parameters.
template <class P, class A>
inline P convert_arg(A&& a) {
  typedef typename std::decay<A>::type D;
  if constexpr (std::is_pointer<P>::value && std::is_integral<D>::value) {
    return a == 0 ? (P) nullptr : (P)(uintptr_t)a;
  } else {
    return std::forward<A>(a);
  }
}

template <class... P, class... A>
inline cudaError_t launch_k(void (*k)(P...), const char* clname, const TuInfo* tu, const LaunchCfg& cfg,
                            A&&... a) {
  static_assert(sizeof...(P) == sizeof...(A), "wrong number of kernel arguments in <<<...>>> launch");
  typedef std::tuple<typename std::decay<P>::type...> Tup;
  Tup args(convert_arg<typename std::decay<P>::type>(std::forward<A>(a))...);
  if (opencl_active()) {
    if (!clname) return record_launch_error(cudaErrorInvalidDeviceFunction);
    ArgPack pack;
    std::apply([&pack](const auto&... v) { (pack.add(v), ...); }, args);
    return gpu_launch(tu, clname, cfg, pack);
  }
  struct Ctx {
    void (*k)(P...);
    Tup* args;
  } ctx{k, &args};
  return cpu_launch(
      tu, cfg, [](void* p) {
        Ctx* c = static_cast<Ctx*>(p);
        std::apply(c->k, *c->args);
      },
      &ctx);
}

// Launch through a function pointer: the kernel is identified by its address at run time.
template <class... P, class... A>
inline cudaError_t launch_dyn(void (*k)(P...), const TuInfo* tu, const LaunchCfg& cfg, A&&... a) {
  return launch_k(k, opencl_active() ? kernel_name_for(reinterpret_cast<const void*>(k)) : nullptr, tu, cfg,
                  std::forward<A>(a)...);
}

template <class F, class... A>
inline cudaError_t launch_generic(const TuInfo* tu, const LaunchCfg& cfg, F f, A&&... a) {
  auto args = std::make_tuple(typename std::decay<A>::type(std::forward<A>(a))...);
  typedef decltype(args) Tup;
  struct Ctx {
    F* f;
    Tup* args;
  } ctx{&f, &args};
  return cpu_launch(
      tu, cfg, [](void* p) {
        Ctx* c = static_cast<Ctx*>(p);
        std::apply(*c->f, *c->args);
      },
      &ctx);
}

// cudaLaunchKernel support: re-type a void** argument array.
template <class... P, size_t... I>
inline cudaError_t launch_void_args_impl(void (*k)(P...), const char* name, const TuInfo* tu, const LaunchCfg& c,
                                         void** args, std::index_sequence<I...>) {
  return launch_k(k, name, tu, c, *static_cast<typename std::decay<P>::type*>(args[I])...);
}
template <class... P>
inline cudaError_t launch_void_args(const void* fn, const char* name, const TuInfo* tu, const LaunchCfg& c,
                                    void** args) {
  void (*k)(P...) = reinterpret_cast<void (*)(P...)>(const_cast<void*>(fn));
  return launch_void_args_impl(k, name, tu, c, args, std::index_sequence_for<P...>{});
}
template <class... P>
inline KernelEntry make_kernel_entry(void (*k)(P...), const char* name) {
  return KernelEntry{reinterpret_cast<const void*>(k), name, &launch_void_args<P...>};
}

}  // namespace pk

// One registration record per translation unit (filled in by code that pkcc
// appends to the end of each translated .cu file).
namespace {
[[maybe_unused]] ::pk::TuInfo pk_tu_info;
}

#endif  // POLYKERNEL_PK_LAUNCH_H
