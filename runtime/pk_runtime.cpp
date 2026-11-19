// PolyKernel: CUDA runtime API implementation (dispatch to CPU or OpenCL back end).
#include "pk_internal.h"

#include <chrono>
#include <cstdarg>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace pk {

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
bool verbose() {
  static int v = [] {
    const char* e = getenv("PK_VERBOSE");
    return (e && *e && *e != '0') ? 1 : 0;
  }();
  return v != 0;
}

void logf(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fputs("polykernel: ", stderr);
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  va_end(ap);
}

size_t env_size(const char* name, size_t def) {
  const char* e = getenv(name);
  if (!e || !*e) return def;
  char* end = nullptr;
  unsigned long long v = strtoull(e, &end, 10);
  if (end && (*end == 'k' || *end == 'K')) v <<= 10;
  if (end && (*end == 'm' || *end == 'M')) v <<= 20;
  if (end && (*end == 'g' || *end == 'G')) v <<= 30;
  return (size_t)v;
}

const char* arch_name() {
#if defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
  return "x86";
#elif defined(__aarch64__) || defined(_M_ARM64)
  return "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
  return "arm";
#elif defined(__riscv) && __riscv_xlen == 64
  return "riscv64";
#elif defined(__riscv)
  return "riscv32";
#elif defined(__s390x__)
  return "s390x";
#elif defined(__powerpc64__) && defined(__LITTLE_ENDIAN__)
  return "ppc64le";
#elif defined(__powerpc64__)
  return "ppc64";
#elif defined(__loongarch64)
  return "loongarch64";
#elif defined(__mips__)
  return "mips";
#elif defined(__wasm__)
  return "wasm";
#else
  return "unknown-cpu";
#endif
}

static thread_local cudaError_t tl_last_error = cudaSuccess;

cudaError_t set_error(cudaError_t e) {
  if (e != cudaSuccess) tl_last_error = e;
  return e;
}
cudaError_t record_launch_error(cudaError_t e) { return set_error(e); }

// ---------------------------------------------------------------------------
// Translation-unit registry and back-end selection
// ---------------------------------------------------------------------------
static std::mutex& tu_mutex() {
  static std::mutex m;
  return m;
}
static TuInfo* g_tus = nullptr;

void register_tu(TuInfo* t) {
  std::lock_guard<std::mutex> l(tu_mutex());
  if (t->registered) return;
  t->registered = true;
  t->next = g_tus;
  g_tus = t;
}
TuInfo* tu_list() { return g_tus; }

const char* kernel_name_for(const void* fn) {
  for (TuInfo* t = g_tus; t; t = t->next)
    for (int i = 0; i < t->nentries; ++i)
      if (t->entries[i].fn == fn) return t->entries[i].name;
  return nullptr;
}

static Backend g_backend = Backend::Cpu;
static std::once_flag g_backend_once;

static bool tus_support(bool ptx, std::string& why) {
  std::set<std::string> kernels;
  int ntu = 0;
  const char* what = ptx ? "PTX" : "OpenCL";
  for (TuInfo* t = g_tus; t; t = t->next) {
    ++ntu;
    if (ptx) {
      for (int i = 0; i < t->nptx_kernels; ++i) kernels.insert(t->ptx_kernels[i]);
    } else {
      for (int i = 0; i < t->nkernels; ++i) kernels.insert(t->kernels[i].name);
    }
  }
  if (ntu == 0) return true;   // no kernels (e.g. deviceQuery): nothing to translate, use the GPU if there is one
  for (TuInfo* t = g_tus; t; t = t->next) {
    if (t->has_generic_launch) {
      why = std::string(t->file ? t->file : "?") +
            ": launches a template kernel or a kernel through a function pointer (CPU-only feature)";
      return false;
    }
    bool ok = ptx ? t->ptx_ok : t->cl_ok;
    const char* reason = ptx ? t->ptx_reason : t->cl_reason;
    if (t->nlaunched > 0 && !ok) {
      why = std::string(t->file ? t->file : "?") + ": no " + what + " translation (" +
            (reason && *reason ? reason : "unknown reason") + ")";
      return false;
    }
    if (t->has_dyn_launch) {
      // a function pointer may target any registered kernel: all of them need a translation
      for (TuInfo* u = g_tus; u; u = u->next) {
        if (u->has_template_kernels) {
          why = std::string(t->file ? t->file : "?") +
                ": launches through a function pointer and the program has template kernels";
          return false;
        }
        for (int k = 0; k < u->nentries; ++k)
          if (!kernels.count(u->entries[k].name)) {
            why = std::string(t->file ? t->file : "?") + ": launches through a function pointer and kernel '" +
                  u->entries[k].name + "' has no " + what + " translation";
            return false;
          }
      }
    }
    for (int i = 0; i < t->nlaunched; ++i) {
      if (!kernels.count(t->launched[i])) {
        why = std::string(t->file ? t->file : "?") + ": kernel '" + t->launched[i] + "' has no " + what +
              " translation";
        return false;
      }
    }
  }
  return true;
}

static void decide_backend() {
  const char* env = getenv("PK_BACKEND");
  std::string want = env && *env ? env : "auto";
  std::string why, why_ptx, why_ocl;
  if (want == "cuda" || want == "nvidia") want = "ptx";
  if (want == "cpu") {
    g_backend = Backend::Cpu;
    why = "PK_BACKEND=cpu";
  } else if (want != "auto" && want != "opencl" && want != "gpu" && want != "ptx") {
    logf("unknown PK_BACKEND='%s' (use auto, cpu, ptx or opencl); using the CPU", want.c_str());
    g_backend = Backend::Cpu;
  } else {
    bool try_ptx = want != "opencl" && getenv("PK_NO_PTX") == nullptr;
    bool try_ocl = want != "ptx";
    if (try_ptx && tus_support(true, why_ptx) && ptx::init(why_ptx)) {
      g_backend = Backend::Ptx;
    } else if (try_ocl && tus_support(false, why_ocl) && ocl::init(want == "opencl", why_ocl)) {
      g_backend = Backend::OpenCL;
    } else {
      g_backend = Backend::Cpu;
      why = try_ptx && !try_ocl ? why_ptx : (try_ocl && !try_ptx ? why_ocl : why_ptx + "; " + why_ocl);
      if (want == "opencl" || want == "ptx")
        logf("%s back end requested but unavailable: %s; using the CPU", want.c_str(), why.c_str());
    }
  }
  if (verbose()) {
    if (g_backend == Backend::Ptx) {
      logf("backend: ptx (%s, driver %d.%d; PolyKernel PTX)", ptx::device_name(), ptx::driver_version() / 1000,
           ptx::driver_version() % 1000 / 10);
    } else if (g_backend == Backend::OpenCL) {
      cudaDeviceProp p;
      ocl::fill_props(&p);
      logf("backend: opencl (%s, %s)%s%s", p.name, ocl::std_option() + 8, why_ptx.empty() ? "" : " - no PTX: ",
           why_ptx.c_str());   // skip "-cl-std="
    } else {
      logf("backend: cpu (%s, %d threads)%s%s", arch_name(), cpu::num_threads(), why.empty() ? "" : " - ",
           why.c_str());
    }
  }
}

Backend backend() {
  std::call_once(g_backend_once, decide_backend);
  return g_backend;
}
bool on_gpu() { return backend() != Backend::Cpu; }
bool opencl_active() { return on_gpu(); }
cudaError_t gpu_launch(const TuInfo* tu, const char* name, const LaunchCfg& cfg, const ArgPack& args) {
  if (backend() == Backend::Ptx) return ptx::launch(tu, name, cfg, args);
  return opencl_launch(tu, name, cfg, args);
}

// ---------------------------------------------------------------------------
// Host-side bookkeeping for the CPU back end
// ---------------------------------------------------------------------------
namespace {
struct HostAlloc {
  size_t size;
  int type;   // cudaMemoryType
};
std::mutex g_mem_m;
std::map<uintptr_t, HostAlloc> g_cpu_allocs;   // CPU-backend device/managed/pinned allocations

void* aligned_alloc_256(size_t n) {
  if (n == 0) n = 1;
#if defined(_WIN32)
  return _aligned_malloc(n, 256);
#else
  void* p = nullptr;
  if (posix_memalign(&p, 256, n) != 0) return nullptr;
  return p;
#endif
}
void aligned_free(void* p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  ::free(p);
#endif
}
cudaError_t cpu_alloc(void** p, size_t n, int type) {
  void* m = aligned_alloc_256(n);
  if (!m) return cudaErrorMemoryAllocation;
  std::lock_guard<std::mutex> l(g_mem_m);
  g_cpu_allocs[(uintptr_t)m] = HostAlloc{n, type};
  *p = m;
  return cudaSuccess;
}
cudaError_t cpu_free(void* p, bool host_only) {
  std::lock_guard<std::mutex> l(g_mem_m);
  auto it = g_cpu_allocs.find((uintptr_t)p);
  if (it == g_cpu_allocs.end()) return host_only ? cudaErrorInvalidValue : cudaErrorInvalidDevicePointer;
  g_cpu_allocs.erase(it);
  aligned_free(p);
  return cudaSuccess;
}
int cpu_pointer_type(const void* p) {
  std::lock_guard<std::mutex> l(g_mem_m);
  auto it = g_cpu_allocs.upper_bound((uintptr_t)p);
  if (it == g_cpu_allocs.begin()) return cudaMemoryTypeUnregistered;
  --it;
  if ((uintptr_t)p < it->first + (it->second.size ? it->second.size : 1)) return it->second.type;
  return cudaMemoryTypeUnregistered;
}

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t physical_memory() {
#if defined(_WIN32)
  MEMORYSTATUSEX s;
  s.dwLength = sizeof s;
  GlobalMemoryStatusEx(&s);
  return (size_t)s.ullTotalPhys;
#else
  long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page <= 0) return (size_t)1 << 32;
  unsigned long long t = (unsigned long long)pages * (unsigned long long)page;
  if (sizeof(size_t) < 8 && t > 0xffffffffull) t = 0xffffffffull;
  return (size_t)t;
#endif
}
size_t available_memory() {
#if defined(_WIN32)
  MEMORYSTATUSEX s;
  s.dwLength = sizeof s;
  GlobalMemoryStatusEx(&s);
  return (size_t)s.ullAvailPhys;
#elif defined(_SC_AVPHYS_PAGES)
  long pages = sysconf(_SC_AVPHYS_PAGES), page = sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page <= 0) return physical_memory() / 2;
  unsigned long long t = (unsigned long long)pages * (unsigned long long)page;
  if (sizeof(size_t) < 8 && t > 0xffffffffull) t = 0xffffffffull;
  return (size_t)t;
#else
  return physical_memory() / 2;
#endif
}

void cpu_fill_props(cudaDeviceProp* p) {
  std::memset(p, 0, sizeof *p);
  snprintf(p->name, sizeof p->name, "PolyKernel CPU (%s, %d threads)", arch_name(), cpu::num_threads());
  p->totalGlobalMem = physical_memory();
  p->sharedMemPerBlock = 48 * 1024;
  p->sharedMemPerBlockOptin = 1024 * 1024;
  p->sharedMemPerMultiprocessor = 1024 * 1024;
  p->regsPerBlock = 65536;
  p->regsPerMultiprocessor = 65536;
  p->warpSize = 32;
  p->memPitch = (size_t)-1 >> 1;
  p->maxThreadsPerBlock = 1024;
  p->maxThreadsDim[0] = 1024;
  p->maxThreadsDim[1] = 1024;
  p->maxThreadsDim[2] = 64;
  p->maxGridSize[0] = 0x7fffffff;
  p->maxGridSize[1] = 65535;
  p->maxGridSize[2] = 65535;
  p->clockRate = 2000000;
  p->memoryClockRate = 2000000;
  p->memoryBusWidth = 64;
  p->totalConstMem = 64 * 1024;
  p->major = 8;
  p->minor = 0;
  p->textureAlignment = 512;
  p->texturePitchAlignment = 32;
  p->multiProcessorCount = cpu::num_threads();
  p->maxThreadsPerMultiProcessor = 2048;
  p->maxBlocksPerMultiProcessor = 32;
  p->integrated = 1;
  p->canMapHostMemory = 1;
  p->unifiedAddressing = 1;
  p->managedMemory = 1;
  p->concurrentManagedAccess = 1;
  p->pageableMemoryAccess = 1;
  p->directManagedMemAccessFromHost = 1;
  p->hostNativeAtomicSupported = 1;
  p->l2CacheSize = 4 * 1024 * 1024;
  p->computeMode = cudaComputeModeDefault;
  p->singleToDoublePrecisionPerfRatio = 2;
  p->hostRegisterSupported = 1;
}
}  // namespace

}  // namespace pk

using namespace pk;

struct CUstream_st {
  unsigned flags;
};
struct CUevent_st {
  unsigned flags;
  bool recorded;
  double t_ms;
};

static cudaError_t device_sync() {
  if (on_gpu()) return dev::synchronize();
  return cudaSuccess;   // CPU launches complete before returning
}

extern "C" {

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------
cudaError_t cudaGetLastError(void) {
  cudaError_t e = tl_last_error;
  tl_last_error = cudaSuccess;
  return e;
}
cudaError_t cudaPeekAtLastError(void) { return tl_last_error; }

#define PK_ERRORS(X)                                                                  \
  X(cudaSuccess, "no error")                                                          \
  X(cudaErrorInvalidValue, "invalid argument")                                        \
  X(cudaErrorMemoryAllocation, "out of memory")                                       \
  X(cudaErrorInitializationError, "initialization error")                             \
  X(cudaErrorCudartUnloading, "driver shutting down")                                 \
  X(cudaErrorInvalidConfiguration, "invalid configuration argument")                  \
  X(cudaErrorInvalidPitchValue, "invalid pitch argument")                             \
  X(cudaErrorInvalidSymbol, "invalid device symbol")                                  \
  X(cudaErrorInvalidHostPointer, "invalid host pointer")                              \
  X(cudaErrorInvalidDevicePointer, "invalid device pointer")                          \
  X(cudaErrorInvalidMemcpyDirection, "invalid copy direction for memcpy")             \
  X(cudaErrorInvalidDeviceFunction, "invalid device function")                        \
  X(cudaErrorNoDevice, "no CUDA-capable device is detected")                          \
  X(cudaErrorInvalidDevice, "invalid device ordinal")                                 \
  X(cudaErrorInvalidKernelImage, "device kernel image is invalid")                    \
  X(cudaErrorNoKernelImageForDevice, "no kernel image is available for execution")    \
  X(cudaErrorInvalidResourceHandle, "invalid resource handle")                        \
  X(cudaErrorSymbolNotFound, "named symbol not found")                                \
  X(cudaErrorNotReady, "device not ready")                                            \
  X(cudaErrorIllegalAddress, "an illegal memory access was encountered")              \
  X(cudaErrorLaunchOutOfResources, "too many resources requested for launch")         \
  X(cudaErrorLaunchFailure, "unspecified launch failure")                             \
  X(cudaErrorNotSupported, "operation not supported")                                 \
  X(cudaErrorPeerAccessUnsupported, "peer access is not supported")                   \
  X(cudaErrorUnknown, "unknown error")

const char* cudaGetErrorString(cudaError_t e) {
#define X(code, text) \
  if (e == code) return text;
  PK_ERRORS(X)
#undef X
  return "unrecognized error code";
}
const char* cudaGetErrorName(cudaError_t e) {
#define X(code, text) \
  if (e == code) return #code;
  PK_ERRORS(X)
#undef X
  return "cudaErrorUnknown";
}

// ---------------------------------------------------------------------------
// Device management (one logical device)
// ---------------------------------------------------------------------------
cudaError_t cudaGetDeviceCount(int* count) {
  if (!count) return set_error(cudaErrorInvalidValue);
  *count = 1;
  return cudaSuccess;
}
cudaError_t cudaGetDevice(int* device) {
  if (!device) return set_error(cudaErrorInvalidValue);
  *device = 0;
  return cudaSuccess;
}
cudaError_t cudaSetDevice(int device) { return device == 0 ? cudaSuccess : set_error(cudaErrorInvalidDevice); }
cudaError_t cudaInitDevice(int device, unsigned int, unsigned int) { return cudaSetDevice(device); }
cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp*) {
  if (!device) return set_error(cudaErrorInvalidValue);
  *device = 0;
  return cudaSuccess;
}
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
  if (!prop) return set_error(cudaErrorInvalidValue);
  if (device != 0) return set_error(cudaErrorInvalidDevice);
  if (on_gpu())
    dev::fill_props(prop);
  else
    cpu_fill_props(prop);
  return cudaSuccess;
}
cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
  if (!value) return set_error(cudaErrorInvalidValue);
  if (device != 0) return set_error(cudaErrorInvalidDevice);
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  int v = 0;
  switch (attr) {
    case cudaDevAttrMaxThreadsPerBlock: v = p.maxThreadsPerBlock; break;
    case cudaDevAttrMaxBlockDimX: v = p.maxThreadsDim[0]; break;
    case cudaDevAttrMaxBlockDimY: v = p.maxThreadsDim[1]; break;
    case cudaDevAttrMaxBlockDimZ: v = p.maxThreadsDim[2]; break;
    case cudaDevAttrMaxGridDimX: v = p.maxGridSize[0]; break;
    case cudaDevAttrMaxGridDimY: v = p.maxGridSize[1]; break;
    case cudaDevAttrMaxGridDimZ: v = p.maxGridSize[2]; break;
    case cudaDevAttrMaxSharedMemoryPerBlock: v = (int)p.sharedMemPerBlock; break;
    case cudaDevAttrTotalConstantMemory: v = (int)p.totalConstMem; break;
    case cudaDevAttrWarpSize: v = p.warpSize; break;
    case cudaDevAttrMaxPitch: v = 0x7fffffff; break;
    case cudaDevAttrMaxRegistersPerBlock: v = p.regsPerBlock; break;
    case cudaDevAttrClockRate: v = p.clockRate; break;
    case cudaDevAttrTextureAlignment: v = (int)p.textureAlignment; break;
    case cudaDevAttrMultiProcessorCount: v = p.multiProcessorCount; break;
    case cudaDevAttrIntegrated: v = p.integrated; break;
    case cudaDevAttrCanMapHostMemory: v = p.canMapHostMemory; break;
    case cudaDevAttrComputeMode: v = p.computeMode; break;
    case cudaDevAttrConcurrentKernels: v = p.concurrentKernels; break;
    case cudaDevAttrMemoryClockRate: v = p.memoryClockRate; break;
    case cudaDevAttrGlobalMemoryBusWidth: v = p.memoryBusWidth; break;
    case cudaDevAttrL2CacheSize: v = p.l2CacheSize; break;
    case cudaDevAttrMaxThreadsPerMultiProcessor: v = p.maxThreadsPerMultiProcessor; break;
    case cudaDevAttrUnifiedAddressing: v = p.unifiedAddressing; break;
    case cudaDevAttrComputeCapabilityMajor: v = p.major; break;
    case cudaDevAttrComputeCapabilityMinor: v = p.minor; break;
    case cudaDevAttrMaxSharedMemoryPerMultiprocessor: v = (int)p.sharedMemPerMultiprocessor; break;
    case cudaDevAttrMaxRegistersPerMultiprocessor: v = p.regsPerMultiprocessor; break;
    case cudaDevAttrManagedMemory: v = p.managedMemory; break;
    case cudaDevAttrPageableMemoryAccess: v = p.pageableMemoryAccess; break;
    case cudaDevAttrConcurrentManagedAccess: v = p.concurrentManagedAccess; break;
    case cudaDevAttrCooperativeLaunch: v = p.cooperativeLaunch; break;
    case cudaDevAttrMaxSharedMemoryPerBlockOptin: v = (int)p.sharedMemPerBlockOptin; break;
    default: v = 0; break;
  }
  *value = v;
  return cudaSuccess;
}
cudaError_t cudaDeviceSynchronize(void) { return set_error(device_sync()); }
cudaError_t cudaThreadSynchronize(void) { return cudaDeviceSynchronize(); }
cudaError_t cudaDeviceReset(void) { return set_error(device_sync()); }
cudaError_t cudaThreadExit(void) { return cudaDeviceReset(); }
cudaError_t cudaSetDeviceFlags(unsigned int) { return cudaSuccess; }
cudaError_t cudaGetDeviceFlags(unsigned int* flags) {
  if (flags) *flags = 0;
  return cudaSuccess;
}
cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache) { return cudaSuccess; }
cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache* c) {
  if (c) *c = cudaFuncCachePreferNone;
  return cudaSuccess;
}
cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig) { return cudaSuccess; }
cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* c) {
  if (c) *c = cudaSharedMemBankSizeFourByte;
  return cudaSuccess;
}
cudaError_t cudaDeviceSetLimit(cudaLimit, size_t) { return cudaSuccess; }
cudaError_t cudaDeviceGetLimit(size_t* v, cudaLimit l) {
  if (!v) return set_error(cudaErrorInvalidValue);
  *v = l == cudaLimitStackSize ? 1024 : (l == cudaLimitMallocHeapSize ? (size_t)8 << 20 : 1 << 20);
  return cudaSuccess;
}
cudaError_t cudaDeviceGetStreamPriorityRange(int* least, int* greatest) {
  if (least) *least = 0;
  if (greatest) *greatest = 0;
  return cudaSuccess;
}
cudaError_t cudaDeviceCanAccessPeer(int* can, int, int) {
  if (can) *can = 0;
  return cudaSuccess;
}
cudaError_t cudaDeviceEnablePeerAccess(int, unsigned int) { return set_error(cudaErrorPeerAccessUnsupported); }
cudaError_t cudaDeviceDisablePeerAccess(int) { return set_error(cudaErrorPeerAccessUnsupported); }
cudaError_t cudaDeviceGetPCIBusId(char* id, int len, int) {
  if (id && len > 0) snprintf(id, (size_t)len, "0000:00:00.0");
  return cudaSuccess;
}
cudaError_t cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr, int src, int dst) {
  if (!value) return set_error(cudaErrorInvalidValue);
  if (src != 0 || dst != 0) return set_error(cudaErrorInvalidDevice);
  *value = 0;
  return cudaSuccess;
}
cudaError_t cudaCtxResetPersistingL2Cache(void) { return cudaSuccess; }
cudaError_t cudaStreamSetAttribute(cudaStream_t, cudaLaunchAttributeID, const cudaLaunchAttributeValue* v) {
  return v ? cudaSuccess : set_error(cudaErrorInvalidValue);
}
cudaError_t cudaStreamGetAttribute(cudaStream_t, cudaLaunchAttributeID, cudaLaunchAttributeValue* v) {
  if (!v) return set_error(cudaErrorInvalidValue);
  std::memset(v, 0, sizeof *v);
  return cudaSuccess;
}
cudaError_t cudaDriverGetVersion(int* v) {
  if (!v) return set_error(cudaErrorInvalidValue);
  *v = backend() == Backend::Ptx ? ptx::driver_version() : CUDART_VERSION;   // the real driver's, on NVIDIA
  return cudaSuccess;
}
cudaError_t cudaRuntimeGetVersion(int* v) {
  if (!v) return set_error(cudaErrorInvalidValue);
  *v = CUDART_VERSION;
  return cudaSuccess;
}
cudaError_t cudaMemGetInfo(size_t* fr, size_t* total) {
  if (on_gpu()) {
    dev::mem_info(fr, total);
  } else {
    if (fr) *fr = available_memory();
    if (total) *total = physical_memory();
  }
  return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------
cudaError_t cudaMalloc(void** p, size_t n) {
  if (!p) return set_error(cudaErrorInvalidValue);
  if (on_gpu()) return set_error(dev::malloc(p, n, false));
  return set_error(cpu_alloc(p, n, cudaMemoryTypeDevice));
}
cudaError_t cudaMallocManaged(void** p, size_t n, unsigned int) {
  if (!p) return set_error(cudaErrorInvalidValue);
  if (on_gpu()) return set_error(dev::malloc(p, n, true));
  return set_error(cpu_alloc(p, n, cudaMemoryTypeManaged));
}
cudaError_t cudaMallocAsync(void** p, size_t n, cudaStream_t) { return cudaMalloc(p, n); }
cudaError_t cudaFreeAsync(void* p, cudaStream_t) { return cudaFree(p); }
cudaError_t cudaFree(void* p) {
  if (!p) return cudaSuccess;
  if (on_gpu()) {
    device_sync();
    return set_error(dev::free(p));
  }
  return set_error(cpu_free(p, false));
}
cudaError_t cudaMallocHost(void** p, size_t n) {
  if (!p) return set_error(cudaErrorInvalidValue);
  if (backend() == Backend::Ptx) return set_error(ptx::host_alloc(p, n));
  return set_error(cpu_alloc(p, n, cudaMemoryTypeHost));
}
cudaError_t cudaHostAlloc(void** p, size_t n, unsigned int) { return cudaMallocHost(p, n); }
cudaError_t cudaFreeHost(void* p) {
  if (!p) return cudaSuccess;
  if (backend() == Backend::Ptx) return set_error(ptx::free(p));
  return set_error(cpu_free(p, true));
}
cudaError_t cudaHostGetDevicePointer(void** d, void* h, unsigned int) {
  if (!d) return set_error(cudaErrorInvalidValue);
  if (backend() == Backend::Ptx) return set_error(ptx::host_device_pointer(d, h));
  if (on_gpu()) return set_error(cudaErrorNotSupported);
  *d = h;
  return cudaSuccess;
}
cudaError_t cudaHostGetFlags(unsigned int* f, void*) {
  if (f) *f = 0;
  return cudaSuccess;
}
cudaError_t cudaHostRegister(void*, size_t, unsigned int) { return cudaSuccess; }
cudaError_t cudaHostUnregister(void*) { return cudaSuccess; }
cudaError_t cudaMallocPitch(void** p, size_t* pitch, size_t w, size_t h) {
  if (!p || !pitch) return set_error(cudaErrorInvalidValue);
  size_t pw = (w + 255) & ~(size_t)255;
  if (pw == 0) pw = 256;
  *pitch = pw;
  return cudaMalloc(p, pw * (h ? h : 1));
}

static cudaError_t do_memcpy(void* dst, const void* src, size_t n) {
  if (n == 0) return cudaSuccess;
  if (!dst || !src) return cudaErrorInvalidValue;
  if (on_gpu()) return dev::memcpy(dst, src, n);
  std::memmove(dst, src, n);
  return cudaSuccess;
}
cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind kind) {
  if ((int)kind < 0 || (int)kind > 4) return set_error(cudaErrorInvalidMemcpyDirection);
  return set_error(do_memcpy(dst, src, n));
}
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind kind, cudaStream_t) {
  return cudaMemcpy(dst, src, n, kind);
}
cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t w, size_t h,
                         cudaMemcpyKind kind) {
  if (w > dpitch || w > spitch) return set_error(cudaErrorInvalidPitchValue);
  if ((int)kind < 0 || (int)kind > 4) return set_error(cudaErrorInvalidMemcpyDirection);
  if (dpitch == w && spitch == w) return set_error(do_memcpy(dst, src, w * h));
  for (size_t r = 0; r < h; ++r) {
    cudaError_t e = do_memcpy((char*)dst + r * dpitch, (const char*)src + r * spitch, w);
    if (e != cudaSuccess) return set_error(e);
  }
  return cudaSuccess;
}
cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t w, size_t h,
                              cudaMemcpyKind kind, cudaStream_t) {
  return cudaMemcpy2D(dst, dpitch, src, spitch, w, h, kind);
}
cudaError_t cudaMemcpyPeer(void* dst, int, const void* src, int, size_t n) {
  return set_error(do_memcpy(dst, src, n));
}
cudaError_t cudaMemcpyPeerAsync(void* dst, int, const void* src, int, size_t n, cudaStream_t) {
  return set_error(do_memcpy(dst, src, n));
}
cudaError_t cudaMemset(void* p, int v, size_t n) {
  if (n == 0) return cudaSuccess;
  if (!p) return set_error(cudaErrorInvalidValue);
  if (on_gpu()) return set_error(dev::memset(p, v, n));
  std::memset(p, v, n);
  return cudaSuccess;
}
cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t) { return cudaMemset(p, v, n); }
cudaError_t cudaMemset2D(void* p, size_t pitch, int v, size_t w, size_t h) {
  for (size_t r = 0; r < h; ++r) {
    cudaError_t e = cudaMemset((char*)p + r * pitch, v, w);
    if (e != cudaSuccess) return e;
  }
  return cudaSuccess;
}
cudaError_t cudaMemset2DAsync(void* p, size_t pitch, int v, size_t w, size_t h, cudaStream_t) {
  return cudaMemset2D(p, pitch, v, w, h);
}
cudaError_t cudaMemPrefetchAsync(const void*, size_t, int, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaMemAdvise(const void*, size_t, cudaMemoryAdvise, int) { return cudaSuccess; }
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* a, const void* p) {
  if (!a) return set_error(cudaErrorInvalidValue);
  int t = cpu_pointer_type(p);
  if (t == cudaMemoryTypeUnregistered && on_gpu()) t = dev::pointer_type(p);
  a->type = (cudaMemoryType)t;
  a->device = 0;
  a->devicePointer = (t == cudaMemoryTypeUnregistered) ? nullptr : const_cast<void*>(p);
  a->hostPointer = (t == cudaMemoryTypeDevice && on_gpu()) ? nullptr : const_cast<void*>(p);
  if (t == cudaMemoryTypeUnregistered) a->hostPointer = const_cast<void*>(p);
  return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Symbols
// ---------------------------------------------------------------------------
static bool find_symbol(const void* sym, const SymbolMeta** out, const TuInfo** tu_out) {
  for (TuInfo* t = tu_list(); t; t = t->next)
    for (int i = 0; i < t->nsymbols; ++i)
      if (t->symbols[i].host_addr == sym) {
        *out = &t->symbols[i];
        *tu_out = t;
        return true;
      }
  return false;
}
cudaError_t pkMemcpyToSymbol(const void* sym, const void* src, size_t n, size_t off, cudaMemcpyKind) {
  if (!sym || (!src && n)) return set_error(cudaErrorInvalidValue);
  if (on_gpu()) {
    const SymbolMeta* s;
    const TuInfo* tu;
    if (!find_symbol(sym, &s, &tu)) return set_error(cudaErrorInvalidSymbol);
    if (off + n > s->size) return set_error(cudaErrorInvalidValue);
    return set_error(dev::to_symbol(s, tu, src, n, off));
  }
  std::memmove((char*)const_cast<void*>(sym) + off, src, n);
  return cudaSuccess;
}
cudaError_t pkMemcpyFromSymbol(void* dst, const void* sym, size_t n, size_t off, cudaMemcpyKind) {
  if (!sym || (!dst && n)) return set_error(cudaErrorInvalidValue);
  if (on_gpu()) {
    const SymbolMeta* s;
    const TuInfo* tu;
    if (!find_symbol(sym, &s, &tu)) return set_error(cudaErrorInvalidSymbol);
    if (off + n > s->size) return set_error(cudaErrorInvalidValue);
    return set_error(dev::from_symbol(s, tu, dst, n, off));
  }
  std::memmove(dst, (const char*)sym + off, n);
  return cudaSuccess;
}
cudaError_t pkGetSymbolAddress(void** p, const void* sym) {
  if (!p) return set_error(cudaErrorInvalidValue);
  if (on_gpu()) {
    const SymbolMeta* s;
    const TuInfo* tu;
    if (!find_symbol(sym, &s, &tu)) return set_error(cudaErrorInvalidSymbol);
    *p = dev::symbol_address(s, tu);
    return *p ? cudaSuccess : set_error(cudaErrorInvalidSymbol);
  }
  *p = const_cast<void*>(sym);
  return cudaSuccess;
}
cudaError_t pkGetSymbolSize(size_t* size, const void* sym) {
  if (!size) return set_error(cudaErrorInvalidValue);
  const SymbolMeta* s;
  const TuInfo* tu;
  if (!find_symbol(sym, &s, &tu)) return set_error(cudaErrorInvalidSymbol);
  *size = s->size;
  return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Streams and events (operations complete in order; launches are synchronous
// on the CPU back end and in-order on a single OpenCL queue)
// ---------------------------------------------------------------------------
cudaError_t cudaStreamCreate(cudaStream_t* s) { return cudaStreamCreateWithFlags(s, 0); }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned int flags) {
  if (!s) return set_error(cudaErrorInvalidValue);
  *s = new CUstream_st{flags};
  return cudaSuccess;
}
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* s, unsigned int flags, int) {
  return cudaStreamCreateWithFlags(s, flags);
}
cudaError_t cudaStreamDestroy(cudaStream_t s) {
  if (s && s != cudaStreamLegacy && s != cudaStreamPerThread) delete s;
  return cudaSuccess;
}
cudaError_t cudaStreamSynchronize(cudaStream_t) { return set_error(device_sync()); }
cudaError_t cudaStreamQuery(cudaStream_t) { return set_error(device_sync()); }
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int) { return cudaSuccess; }
cudaError_t cudaStreamAddCallback(cudaStream_t s, cudaStreamCallback_t cb, void* data, unsigned int) {
  cudaError_t e = device_sync();
  if (cb) cb(s, e, data);
  return set_error(e);
}
cudaError_t cudaLaunchHostFunc(cudaStream_t, cudaHostFn_t fn, void* data) {
  cudaError_t e = device_sync();
  if (fn) fn(data);
  return set_error(e);
}
cudaError_t cudaEventCreate(cudaEvent_t* e) { return cudaEventCreateWithFlags(e, 0); }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned int flags) {
  if (!e) return set_error(cudaErrorInvalidValue);
  *e = new CUevent_st{flags, false, 0.0};
  return cudaSuccess;
}
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t) {
  if (!e) return set_error(cudaErrorInvalidResourceHandle);
  cudaError_t err = device_sync();
  e->recorded = true;
  e->t_ms = now_ms();
  return set_error(err);
}
cudaError_t cudaEventSynchronize(cudaEvent_t e) {
  if (!e) return set_error(cudaErrorInvalidResourceHandle);
  return set_error(device_sync());
}
cudaError_t cudaEventQuery(cudaEvent_t e) {
  if (!e) return set_error(cudaErrorInvalidResourceHandle);
  return cudaSuccess;
}
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
  if (!ms || !a || !b) return set_error(cudaErrorInvalidValue);
  if ((a->flags | b->flags) & cudaEventDisableTiming) return set_error(cudaErrorInvalidResourceHandle);
  if (!a->recorded || !b->recorded) return set_error(cudaErrorInvalidResourceHandle);
  *ms = (float)(b->t_ms - a->t_ms);
  return cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t e) {
  delete e;
  return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Execution control
// ---------------------------------------------------------------------------
cudaError_t cudaLaunchKernel(const void* func, dim3 g, dim3 b, void** args, size_t shmem, cudaStream_t s) {
  for (TuInfo* t = tu_list(); t; t = t->next)
    for (int i = 0; i < t->nentries; ++i)
      if (t->entries[i].fn == func) {
        LaunchCfg cfg(g, b, shmem, s);
        return t->entries[i].launcher(func, t->entries[i].name, t, cfg, args);
      }
  return set_error(cudaErrorInvalidDeviceFunction);
}
cudaError_t cudaFuncSetCacheConfig(const void*, cudaFuncCache) { return cudaSuccess; }
cudaError_t cudaFuncSetAttribute(const void*, cudaFuncAttribute, int) { return cudaSuccess; }
cudaError_t cudaFuncSetSharedMemConfig(const void*, cudaSharedMemConfig) { return cudaSuccess; }
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, const void*) {
  if (!a) return set_error(cudaErrorInvalidValue);
  std::memset(a, 0, sizeof *a);
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  a->maxThreadsPerBlock = p.maxThreadsPerBlock;
  a->numRegs = 32;
  a->ptxVersion = p.major * 10 + p.minor;
  a->binaryVersion = a->ptxVersion;
  a->maxDynamicSharedSizeBytes = (int)p.sharedMemPerBlock;
  return cudaSuccess;
}
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, const void*, int bs, size_t) {
  if (!n) return set_error(cudaErrorInvalidValue);
  *n = bs > 0 ? (2048 / bs > 0 ? 2048 / bs : 1) : 1;
  return cudaSuccess;
}
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* n, const void* f, int bs, size_t sm,
                                                                   unsigned int) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(n, f, bs, sm);
}
cudaError_t cudaProfilerStart(void) { return cudaSuccess; }
cudaError_t cudaProfilerStop(void) { return cudaSuccess; }

}  // extern "C"
