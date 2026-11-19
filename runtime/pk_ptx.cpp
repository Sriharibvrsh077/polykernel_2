// PolyKernel PTX back end: runs the PTX that PolyKernel's own compiler (tools/pk_ptx.py)
// generates on NVIDIA GPUs through the CUDA driver API (libcuda, part of the GPU driver).
// No CUDA toolkit is needed. The few driver declarations below are written from NVIDIA's
// public driver API reference; device pointers are real GPU addresses (unified addressing).
#include "pk_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pk {
namespace ptx {

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;

enum {
  CU_JIT_INFO_LOG_BUFFER = 3,
  CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
  CU_JIT_ERROR_LOG_BUFFER = 5,
  CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
  CU_JIT_LOG_VERBOSE = 12,
  CU_MEMHOSTALLOC_PORTABLE = 1,
  CU_MEMHOSTALLOC_DEVICEMAP = 2,
  CU_MEM_ATTACH_GLOBAL = 1,
};

#if defined(_WIN32)
#define PK_CUAPI __stdcall
#else
#define PK_CUAPI
#endif

#define PK_CU_FUNCS(X)                                                                                       \
  X(cuInit, (unsigned))                                                                                      \
  X(cuDriverGetVersion, (int*))                                                                              \
  X(cuDeviceGetCount, (int*))                                                                                \
  X(cuDeviceGet, (CUdevice*, int))                                                                           \
  X(cuDeviceGetName, (char*, int, CUdevice))                                                                 \
  X(cuDeviceGetAttribute, (int*, int, CUdevice))                                                             \
  X(cuDeviceTotalMem_v2, (size_t*, CUdevice))                                                                \
  X(cuDevicePrimaryCtxRetain, (CUcontext*, CUdevice))                                                        \
  X(cuCtxSetCurrent, (CUcontext))                                                                            \
  X(cuCtxSynchronize, ())                                                                                    \
  X(cuModuleLoadDataEx, (CUmodule*, const void*, unsigned, int*, void**))                                    \
  X(cuModuleGetFunction, (CUfunction*, CUmodule, const char*))                                               \
  X(cuModuleGetGlobal_v2, (CUdeviceptr*, size_t*, CUmodule, const char*))                                    \
  X(cuMemAlloc_v2, (CUdeviceptr*, size_t))                                                                   \
  X(cuMemAllocManaged, (CUdeviceptr*, size_t, unsigned))                                                     \
  X(cuMemFree_v2, (CUdeviceptr))                                                                             \
  X(cuMemHostAlloc, (void**, size_t, unsigned))                                                              \
  X(cuMemFreeHost, (void*))                                                                                  \
  X(cuMemHostGetDevicePointer_v2, (CUdeviceptr*, void*, unsigned))                                           \
  X(cuMemcpy, (CUdeviceptr, CUdeviceptr, size_t))                                                            \
  X(cuMemsetD8_v2, (CUdeviceptr, unsigned char, size_t))                                                     \
  X(cuMemGetInfo_v2, (size_t*, size_t*))                                                                     \
  X(cuLaunchKernel, (CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, \
                     void**, void**))                                                                        \
  X(cuGetErrorString, (CUresult, const char**))

#define X(name, args) typedef CUresult(PK_CUAPI* name##_t) args; static name##_t p_##name = nullptr;
PK_CU_FUNCS(X)
#undef X

struct Tu {
  CUmodule mod = nullptr;
  std::map<std::string, CUfunction> funcs;
  std::map<std::string, CUdeviceptr> syms;
};

struct State {
  std::mutex m;
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  char name[256] = {0};
  char label[300] = {0};   // name and compute capability, for PK_VERBOSE
  int cc_major = 0, cc_minor = 0;
  std::map<uintptr_t, std::pair<size_t, int>> allocs;   // base -> (size, cudaMemoryType)
} g;

thread_local bool tl_ctx = false;

static void ensure_ctx() {
  if (!tl_ctx && g.ctx) {
    p_cuCtxSetCurrent(g.ctx);
    tl_ctx = true;
  }
}

static const char* cu_err(CUresult r) {
  const char* s = nullptr;
  if (p_cuGetErrorString && p_cuGetErrorString(r, &s) == 0 && s) return s;
  return "unknown CUDA driver error";
}

// Driver and runtime error codes share their numeric values (public API reference).
static cudaError_t map_err(CUresult r) {
  if (r == 0) return cudaSuccess;
  if (r == 2) return cudaErrorMemoryAllocation;
  return static_cast<cudaError_t>(r);
}

static void* load_driver(std::string& why) {
  std::vector<std::string> names;
  if (const char* e = getenv("PK_CUDA_DRIVER_LIBRARY")) names.push_back(e);
#if defined(_WIN32)
  names.push_back("nvcuda.dll");
  for (const std::string& n : names) {
    HMODULE h = LoadLibraryA(n.c_str());
    if (h) return reinterpret_cast<void*>(h);
  }
#else
  names.push_back("libcuda.so.1");
  names.push_back("libcuda.so");
  for (const std::string& n : names) {
    void* h = dlopen(n.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (h) return h;
  }
#endif
  why = "no NVIDIA driver (libcuda) found";
  return nullptr;
}

static void* sym(void* h, const char* n) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(h), n));
#else
  return dlsym(h, n);
#endif
}

static bool load_module(TuInfo* t, std::string& why) {
  Tu* T = new Tu;
  std::vector<char> elog(16384, 0), ilog(16384, 0);
  // with PK_VERBOSE the JIT also reports registers, spills and shared memory per kernel
  int opts[5] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_INFO_LOG_BUFFER,
                 CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_LOG_VERBOSE};
  void* vals[5] = {elog.data(), reinterpret_cast<void*>(static_cast<uintptr_t>(elog.size())), ilog.data(),
                   reinterpret_cast<void*>(static_cast<uintptr_t>(ilog.size())), reinterpret_cast<void*>(1)};
  CUresult r = p_cuModuleLoadDataEx(&T->mod, t->ptx_src, verbose() ? 5 : 4, opts, vals);
  if (r != 0) {
    why = std::string(t->file ? t->file : "?") + ": the driver rejected the PTX (" + cu_err(r) + ")";
    logf("PTX JIT failed for %s:\n%s", t->file ? t->file : "?", elog.data());
    delete T;
    return false;
  }
  if (verbose() && ilog[0]) logf("PTX JIT %s: %s", t->file ? t->file : "?", ilog.data());
  // __device__/__constant__ variables: start from the host copy (it holds the initial value)
  for (int i = 0; i < t->nsymbols; ++i) {
    const SymbolMeta& sm = t->symbols[i];
    CUdeviceptr d = 0;
    size_t sz = 0;
    std::string n = std::string("pk_g_") + sm.name;
    if (p_cuModuleGetGlobal_v2(&d, &sz, T->mod, n.c_str()) != 0) continue;
    T->syms[sm.name] = d;
    p_cuMemcpy(d, reinterpret_cast<CUdeviceptr>(sm.host_addr), sm.size < sz ? sm.size : sz);
  }
  t->backend_data = T;
  return true;
}

bool init(std::string& why) {
  void* h = load_driver(why);
  if (!h) return false;
  bool ok = true;
#define X(name, args)                                               \
  p_##name = reinterpret_cast<name##_t>(sym(h, #name));             \
  if (!p_##name) {                                                  \
    ok = false;                                                     \
    why = std::string("the NVIDIA driver lacks ") + #name;           \
  }
  PK_CU_FUNCS(X)
#undef X
  if (!ok) return false;
  CUresult r = p_cuInit(0);
  if (r != 0) {
    why = std::string("cuInit failed (") + cu_err(r) + ")";
    return false;
  }
  int n = 0;
  if (p_cuDeviceGetCount(&n) != 0 || n == 0) {
    why = "no NVIDIA GPU";
    return false;
  }
  int idx = 0;
  if (const char* e = getenv("PK_CUDA_DEVICE")) idx = atoi(e);
  if (idx < 0 || idx >= n) idx = 0;
  if (p_cuDeviceGet(&g.dev, idx) != 0 || p_cuDevicePrimaryCtxRetain(&g.ctx, g.dev) != 0) {
    why = "cannot create a CUDA context";
    return false;
  }
  p_cuCtxSetCurrent(g.ctx);
  tl_ctx = true;
  p_cuDeviceGetName(g.name, sizeof g.name - 1, g.dev);
  p_cuDeviceGetAttribute(&g.cc_major, 75, g.dev);
  p_cuDeviceGetAttribute(&g.cc_minor, 76, g.dev);
  snprintf(g.label, sizeof g.label, "%s, sm_%d%d", g.name, g.cc_major, g.cc_minor);
  for (TuInfo* t = tu_list(); t; t = t->next) {
    if (!t->ptx_src || !t->ptx_ok) continue;
    if (!load_module(t, why)) {
      for (TuInfo* u = tu_list(); u; u = u->next) u->backend_data = nullptr;   // leave nothing for OpenCL
      return false;
    }
  }
  // device printf output is flushed at synchronisation points; make sure the last kernels' output appears
  atexit([] {
    if (g.ctx && p_cuCtxSetCurrent && p_cuCtxSynchronize) {
      p_cuCtxSetCurrent(g.ctx);
      p_cuCtxSynchronize();
    }
  });
  return true;
}

const char* device_name() { return g.label; }

int driver_version() {
  int v = 0;
  if (p_cuDriverGetVersion) p_cuDriverGetVersion(&v);
  return v;
}

int attribute(cudaDeviceAttr a, int* ok) {
  ensure_ctx();
  int v = 0;
  CUresult r = p_cuDeviceGetAttribute(&v, static_cast<int>(a), g.dev);
  if (ok) *ok = r == 0;
  return v;
}

void fill_props(cudaDeviceProp* p) {
  ensure_ctx();
  std::memset(p, 0, sizeof *p);
  snprintf(p->name, sizeof p->name, "%s", g.name);
  size_t total = 0;
  p_cuDeviceTotalMem_v2(&total, g.dev);
  p->totalGlobalMem = total;
  auto A = [](int attr) {
    int v = 0;
    p_cuDeviceGetAttribute(&v, attr, g.dev);
    return v;
  };
  p->maxThreadsPerBlock = A(1);
  p->maxThreadsDim[0] = A(2);
  p->maxThreadsDim[1] = A(3);
  p->maxThreadsDim[2] = A(4);
  p->maxGridSize[0] = A(5);
  p->maxGridSize[1] = A(6);
  p->maxGridSize[2] = A(7);
  p->sharedMemPerBlock = A(8);
  p->totalConstMem = A(9);
  p->warpSize = A(10);
  p->memPitch = A(11);
  p->regsPerBlock = A(12);
  p->clockRate = A(13);
  p->textureAlignment = A(14);
  p->deviceOverlap = A(15);
  p->multiProcessorCount = A(16);
  p->kernelExecTimeoutEnabled = A(17);
  p->integrated = A(18);
  p->canMapHostMemory = A(19);
  p->computeMode = A(20);
  p->concurrentKernels = A(31);
  p->ECCEnabled = A(32);
  p->pciBusID = A(33);
  p->pciDeviceID = A(34);
  p->tccDriver = A(35);
  p->memoryClockRate = A(36);
  p->memoryBusWidth = A(37);
  p->l2CacheSize = A(38);
  p->maxThreadsPerMultiProcessor = A(39);
  p->asyncEngineCount = A(40);
  p->unifiedAddressing = A(41);
  p->pciDomainID = A(50);
  p->major = A(75);
  p->minor = A(76);
  p->streamPrioritiesSupported = A(78);
  p->globalL1CacheSupported = A(79);
  p->localL1CacheSupported = A(80);
  p->sharedMemPerMultiprocessor = A(81);
  p->regsPerMultiprocessor = A(82);
  p->managedMemory = A(83);
  p->isMultiGpuBoard = A(84);
  p->multiGpuBoardGroupID = A(85);
  p->pageableMemoryAccess = A(88);
  p->concurrentManagedAccess = A(89);
  p->cooperativeLaunch = A(95);
  p->sharedMemPerBlockOptin = A(97);
}

static void track(void* p, size_t n, int type) {
  std::lock_guard<std::mutex> lk(g.m);
  g.allocs[reinterpret_cast<uintptr_t>(p)] = std::make_pair(n, type);
}

cudaError_t malloc(void** p, size_t n, bool managed) {
  ensure_ctx();
  CUdeviceptr d = 0;
  CUresult r = managed ? p_cuMemAllocManaged(&d, n ? n : 1, CU_MEM_ATTACH_GLOBAL) : p_cuMemAlloc_v2(&d, n ? n : 1);
  if (r != 0) {
    *p = nullptr;
    return map_err(r);
  }
  *p = reinterpret_cast<void*>(d);
  track(*p, n, managed ? cudaMemoryTypeManaged : cudaMemoryTypeDevice);
  return cudaSuccess;
}

cudaError_t host_alloc(void** p, size_t n) {
  ensure_ctx();
  CUresult r = p_cuMemHostAlloc(p, n ? n : 1, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
  if (r != 0) {
    *p = nullptr;
    return map_err(r);
  }
  track(*p, n, cudaMemoryTypeHost);
  return cudaSuccess;
}

cudaError_t host_device_pointer(void** d, void* h) {
  ensure_ctx();
  CUdeviceptr dp = 0;
  CUresult r = p_cuMemHostGetDevicePointer_v2(&dp, h, 0);
  if (r != 0) return map_err(r);
  *d = reinterpret_cast<void*>(dp);
  return cudaSuccess;
}

cudaError_t free(void* p) {
  ensure_ctx();
  int type = cudaMemoryTypeDevice;
  {
    std::lock_guard<std::mutex> lk(g.m);
    auto it = g.allocs.find(reinterpret_cast<uintptr_t>(p));
    if (it == g.allocs.end()) return cudaErrorInvalidDevicePointer;
    type = it->second.second;
    g.allocs.erase(it);
  }
  CUresult r = type == cudaMemoryTypeHost ? p_cuMemFreeHost(p) : p_cuMemFree_v2(reinterpret_cast<CUdeviceptr>(p));
  return map_err(r);
}

cudaError_t memcpy(void* dst, const void* src, size_t n) {
  if (n == 0) return cudaSuccess;
  ensure_ctx();
  return map_err(p_cuMemcpy(reinterpret_cast<CUdeviceptr>(dst), reinterpret_cast<CUdeviceptr>(src), n));
}

cudaError_t memset(void* p, int v, size_t n) {
  if (n == 0) return cudaSuccess;
  ensure_ctx();
  return map_err(p_cuMemsetD8_v2(reinterpret_cast<CUdeviceptr>(p), static_cast<unsigned char>(v), n));
}

cudaError_t synchronize() {
  ensure_ctx();
  return map_err(p_cuCtxSynchronize());
}

int pointer_type(const void* p) {
  std::lock_guard<std::mutex> lk(g.m);
  uintptr_t a = reinterpret_cast<uintptr_t>(p);
  auto it = g.allocs.upper_bound(a);
  if (it == g.allocs.begin()) return 0;
  --it;
  if (a < it->first + (it->second.first ? it->second.first : 1)) return it->second.second;
  return 0;
}

void mem_info(size_t* free_b, size_t* total_b) {
  ensure_ctx();
  size_t f = 0, t = 0;
  p_cuMemGetInfo_v2(&f, &t);
  if (free_b) *free_b = f;
  if (total_b) *total_b = t;
}

static CUdeviceptr symbol_dptr(const SymbolMeta* s, const TuInfo* tu) {
  Tu* T = static_cast<Tu*>(tu->backend_data);
  if (!T) return 0;
  auto it = T->syms.find(s->name);
  return it == T->syms.end() ? 0 : it->second;
}

cudaError_t to_symbol(const SymbolMeta* s, const TuInfo* tu, const void* src, size_t n, size_t off) {
  CUdeviceptr d = symbol_dptr(s, tu);
  if (!d) return cudaErrorInvalidSymbol;
  return memcpy(reinterpret_cast<void*>(d + off), src, n);
}

cudaError_t from_symbol(const SymbolMeta* s, const TuInfo* tu, void* dst, size_t n, size_t off) {
  CUdeviceptr d = symbol_dptr(s, tu);
  if (!d) return cudaErrorInvalidSymbol;
  return memcpy(dst, reinterpret_cast<const void*>(d + off), n);
}

void* symbol_address(const SymbolMeta* s, const TuInfo* tu) { return reinterpret_cast<void*>(symbol_dptr(s, tu)); }

static CUfunction find_function(Tu* T, const char* name) {
  auto it = T->funcs.find(name);
  if (it != T->funcs.end()) return it->second;
  CUfunction f = nullptr;
  std::string ename = std::string("pk_k_") + name;
  if (p_cuModuleGetFunction(&f, T->mod, ename.c_str()) != 0) f = nullptr;
  T->funcs[name] = f;
  return f;
}

cudaError_t launch(const TuInfo* tu, const char* name, const LaunchCfg& c, const ArgPack& a) {
  ensure_ctx();
  // The kernel is normally in the launching file's module; a kernel defined in another file
  // (declared extern here) is looked up in the other modules.
  CUfunction f = nullptr;
  {
    std::lock_guard<std::mutex> lk(g.m);
    if (Tu* T = static_cast<Tu*>(tu->backend_data)) f = find_function(T, name);
    for (const TuInfo* t = tu_list(); t && !f; t = t->next)
      if (t != tu && t->backend_data) f = find_function(static_cast<Tu*>(t->backend_data), name);
  }
  if (!f) {
    logf("kernel '%s' is not in any PTX module", name);
    return record_launch_error(cudaErrorInvalidDeviceFunction);
  }
  if (a.overflow) return record_launch_error(cudaErrorInvalidValue);
  uint64_t vals[ArgPack::kMax];
  void* params[ArgPack::kMax];
  for (int i = 0; i < a.n; ++i) {
    const ArgPack::Arg& x = a.args[i];
    switch (x.kind) {
      case ArgPack::Ptr:
        vals[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(x.data));
        params[i] = &vals[i];
        break;
      case ArgPack::Int:
        vals[i] = x.bits;
        params[i] = &vals[i];
        break;
      case ArgPack::Flt: {
        double d;
        std::memcpy(&d, &x.bits, sizeof d);
        float fl = static_cast<float>(d);
        vals[i] = 0;
        std::memcpy(&vals[i], &fl, sizeof fl);
        params[i] = &vals[i];
        break;
      }
      case ArgPack::Dbl:
        vals[i] = x.bits;
        params[i] = &vals[i];
        break;
      default:
        params[i] = const_cast<void*>(x.data);
        break;
    }
  }
  CUresult r = p_cuLaunchKernel(f, c.grid.x, c.grid.y, c.grid.z, c.block.x, c.block.y, c.block.z,
                                static_cast<unsigned>(c.shmem), nullptr, a.n ? params : nullptr, nullptr);
  if (r != 0) {
    if (verbose()) logf("kernel '%s': launch failed (%s)", name, cu_err(r));
    return record_launch_error(map_err(r));
  }
  if (getenv("PK_SYNC_LAUNCH")) {
    r = p_cuCtxSynchronize();
    if (r != 0) {
      logf("kernel '%s' failed: %s", name, cu_err(r));
      return record_launch_error(map_err(r));
    }
  }
  return cudaSuccess;
}

}  // namespace ptx
}  // namespace pk
