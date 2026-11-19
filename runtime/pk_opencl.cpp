// PolyKernel OpenCL back end: runs translated kernels on any OpenCL 1.2+ device
// (AMD, Intel, ARM Mali, Qualcomm Adreno, Apple, PoCL, ...).
//
//  * libOpenCL is loaded with dlopen/LoadLibrary at first use: binaries do not
//    depend on OpenCL and simply fall back to the CPU back end without it.
//  * Device memory: cudaMalloc returns an address inside a reserved, never
//    mapped range of the host address space; the runtime maps such "device
//    addresses" (including pointer arithmetic on them) to (cl_mem, offset).
//    Each pointer kernel parameter is passed as buffer + byte offset.
//  * Managed memory: a host allocation mirrored by a cl_mem, uploaded before a
//    launch and downloaded at the next synchronization point.
#define CL_TARGET_OPENCL_VERSION 300
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/cl.h"
#include "pk_internal.h"

#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dirent.h>
#include <dlfcn.h>
#include <sys/mman.h>
#endif

namespace pk {
namespace ocl {
namespace {

#define PK_CL_FUNCS(X)                                                                                     \
  X(clGetPlatformIDs) X(clGetPlatformInfo) X(clGetDeviceIDs) X(clGetDeviceInfo) X(clCreateContext)       \
  X(clCreateCommandQueue) X(clCreateBuffer) X(clReleaseMemObject) X(clEnqueueWriteBuffer)                \
  X(clEnqueueReadBuffer) X(clEnqueueCopyBuffer) X(clEnqueueFillBuffer) X(clCreateProgramWithSource)       \
  X(clBuildProgram) X(clGetProgramBuildInfo) X(clCreateKernel) X(clSetKernelArg) X(clEnqueueNDRangeKernel) \
  X(clFinish) X(clGetKernelWorkGroupInfo) X(clReleaseKernel) X(clReleaseProgram)

#define X(f) decltype(&::f) p_##f = nullptr;
PK_CL_FUNCS(X)
#undef X

struct Program {
  cl_program prog = nullptr;
  std::map<std::string, cl_kernel> kernels;
  std::map<std::string, std::vector<std::pair<char, int>>> sigs;   // kind ('P','I','F','S'), size
  std::map<std::string, std::vector<std::string>> ksyms;   // kernel -> device variables it takes as hidden args
  std::map<std::string, void*> symaddr;                    // device variable -> its buffer (device address)
};

struct Alloc {
  uintptr_t base;   // device address (or host address for managed memory)
  size_t size;
  cl_mem mem;
  bool managed;
  bool dev_newer;   // managed: device copy is newer than the host copy
  size_t va_off, va_len;
};

struct State {
  cl_platform_id plat = nullptr;
  cl_device_id dev = nullptr;
  cl_context ctx = nullptr;
  cl_command_queue q = nullptr;
  std::string dev_name, std_opt;
  size_t max_wg = 256;
  size_t max_item[3] = {256, 256, 64};
  cl_ulong global_mem = 0, local_mem = 0, max_alloc = 0, const_mem = 0;
  cl_uint cus = 1, clock_mhz = 0;
  cl_bool unified = CL_FALSE;
  // fake device address range
  uintptr_t va_base = 0;
  size_t va_size = 0, va_top = 0;
  std::map<size_t, size_t> va_free;   // offset -> length
  std::map<uintptr_t, Alloc> allocs;
  std::set<uintptr_t> pending_managed;
  std::mutex m;
} g;

std::string cl_str_dev(cl_device_id d, cl_device_info what) {
  size_t n = 0;
  if (p_clGetDeviceInfo(d, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) return "";
  std::string s(n, '\0');
  p_clGetDeviceInfo(d, what, n, &s[0], nullptr);
  while (!s.empty() && (s.back() == '\0' || s.back() == ' ')) s.pop_back();
  return s;
}

#if !defined(_WIN32) && !defined(__APPLE__)
// Without libOpenCL (the Khronos ICD loader) a vendor driver can still be used directly: every
// OpenCL object starts with a pointer to the driver's cl_khr_icd dispatch table, whose layout is a
// fixed ABI. This lets PolyKernel use e.g. libnvidia-opencl.so.1 in containers without ocl-icd.
typedef cl_int(CL_API_CALL* IcdGetPlatforms)(cl_uint, cl_platform_id*, cl_uint*);
IcdGetPlatforms g_icd_platforms = nullptr;
cl_int CL_API_CALL icd_get_platform_ids(cl_uint n, cl_platform_id* p, cl_uint* np) { return g_icd_platforms(n, p, np); }

#define PK_ICD_SLOTS(X)                                                                                     \
  X(clGetPlatformInfo, 1) X(clGetDeviceIDs, 2) X(clGetDeviceInfo, 3) X(clCreateContext, 4)                  \
  X(clCreateCommandQueue, 9) X(clCreateBuffer, 14) X(clReleaseMemObject, 18) X(clCreateProgramWithSource, 26) \
  X(clReleaseProgram, 29) X(clBuildProgram, 30) X(clGetProgramBuildInfo, 33) X(clCreateKernel, 34)          \
  X(clReleaseKernel, 37) X(clSetKernelArg, 38) X(clGetKernelWorkGroupInfo, 40) X(clFinish, 47)              \
  X(clEnqueueReadBuffer, 48) X(clEnqueueWriteBuffer, 49) X(clEnqueueCopyBuffer, 50)                         \
  X(clEnqueueNDRangeKernel, 59) X(clEnqueueFillBuffer, 102)

bool load_vendor_icd(std::string& why) {
  std::vector<std::string> libs;
  if (const char* e = getenv("PK_OPENCL_VENDOR_LIBRARY")) libs.push_back(e);
  const char* vdir = getenv("OCL_ICD_VENDORS");
  std::string dir = vdir && *vdir ? vdir : "/etc/OpenCL/vendors";
  std::vector<std::string> listed;
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* de = readdir(d)) {
      std::string fn = de->d_name;
      if (fn.size() < 5 || fn.compare(fn.size() - 4, 4, ".icd") != 0) continue;
      if (FILE* f = fopen((dir + "/" + fn).c_str(), "r")) {
        char line[1024] = {0};
        if (fgets(line, sizeof line, f)) {
          std::string l = line;
          while (!l.empty() && (l.back() == '\n' || l.back() == '\r' || l.back() == ' ')) l.pop_back();
          if (!l.empty()) listed.push_back(l);
        }
        fclose(f);
      }
    }
    closedir(d);
  }
  // GPU drivers before CPU runtimes
  for (const std::string& l : listed)
    if (l.find("pocl") == std::string::npos) libs.push_back(l);
  for (const char* n : {"libnvidia-opencl.so.1", "libamdocl64.so", "libigdrcl.so", "libmali.so", "libOpenCL_adreno.so"})
    libs.push_back(n);
  for (const std::string& l : listed)
    if (l.find("pocl") != std::string::npos) libs.push_back(l);
  libs.push_back("libpocl.so.2");
  for (const std::string& n : libs) {
    void* h = dlopen(n.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) continue;
    IcdGetPlatforms getp = reinterpret_cast<IcdGetPlatforms>(dlsym(h, "clIcdGetPlatformIDsKHR"));
    if (!getp) {
      typedef void*(CL_API_CALL * Gefa)(const char*);
      Gefa gefa = reinterpret_cast<Gefa>(dlsym(h, "clGetExtensionFunctionAddress"));
      if (gefa) getp = reinterpret_cast<IcdGetPlatforms>(gefa("clIcdGetPlatformIDsKHR"));
    }
    cl_uint np = 0;
    if (!getp || getp(0, nullptr, &np) != CL_SUCCESS || np == 0) continue;
    std::vector<cl_platform_id> ps(np);
    if (getp(np, ps.data(), nullptr) != CL_SUCCESS || !ps[0]) continue;
    void* const* tbl = *reinterpret_cast<void* const* const*>(ps[0]);
    g_icd_platforms = getp;
    p_clGetPlatformIDs = &icd_get_platform_ids;
#define X(f, slot) p_##f = reinterpret_cast<decltype(p_##f)>(tbl[slot]);
    PK_ICD_SLOTS(X)
#undef X
    bool ok = true;
#define X(f) ok = ok && p_##f != nullptr;
    PK_CL_FUNCS(X)
#undef X
    if (ok) {
      if (verbose()) logf("OpenCL: using the driver %s directly (cl_khr_icd dispatch, no libOpenCL)", n.c_str());
      return true;
    }
  }
  why = "no OpenCL library found (install an OpenCL ICD loader, e.g. ocl-icd-libopencl1, or a GPU driver)";
  return false;
}
#endif

bool load_library(std::string& why) {
#if !defined(_WIN32) && !defined(__APPLE__)
  if (const char* e = getenv("PK_OPENCL_ICD"))   // 1: skip libOpenCL and talk to the driver directly
    if (*e == '1') return load_vendor_icd(why);
#endif
  std::vector<std::string> names;
  if (const char* e = getenv("PK_OPENCL_LIBRARY")) names.push_back(e);
#if defined(_WIN32)
  names.push_back("OpenCL.dll");
#elif defined(__APPLE__)
  names.push_back("/System/Library/Frameworks/OpenCL.framework/OpenCL");
#else
  names.push_back("libOpenCL.so.1");
  names.push_back("libOpenCL.so");
#endif
  for (const std::string& n : names) {
#if defined(_WIN32)
    HMODULE h = LoadLibraryA(n.c_str());
    if (!h) continue;
#define X(f) p_##f = reinterpret_cast<decltype(p_##f)>(reinterpret_cast<void*>(GetProcAddress(h, #f)));
#else
    void* h = dlopen(n.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) continue;
#define X(f) p_##f = reinterpret_cast<decltype(p_##f)>(dlsym(h, #f));
#endif
    PK_CL_FUNCS(X)
#undef X
    bool ok = true;
#define X(f) ok = ok && p_##f != nullptr;
    PK_CL_FUNCS(X)
#undef X
    if (ok) return true;
    why = "the OpenCL library " + n + " lacks OpenCL 1.2 entry points";
    return false;
  }
#if !defined(_WIN32) && !defined(__APPLE__)
  return load_vendor_icd(why);
#else
  why = "no OpenCL library found";
  return false;
#endif
}

bool reserve_address_range() {
  size_t sizes64[] = {(size_t)1 << 36, (size_t)1 << 34, (size_t)1 << 32, (size_t)1 << 30};
  size_t sizes32[] = {(size_t)1 << 28, (size_t)1 << 26};
  size_t* sizes = sizeof(void*) == 8 ? sizes64 : sizes32;
  int count = sizeof(void*) == 8 ? 4 : 2;
  for (int i = 0; i < count; ++i) {
#if defined(_WIN32)
    void* p = VirtualAlloc(nullptr, sizes[i], MEM_RESERVE, PAGE_NOACCESS);
    if (!p) continue;
#else
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void* p = mmap(nullptr, sizes[i], PROT_NONE, flags, -1, 0);
    if (p == MAP_FAILED) continue;
#endif
    g.va_base = (uintptr_t)p;
    g.va_size = sizes[i];
    return true;
  }
  return false;
}

bool va_alloc(size_t n, size_t* off, size_t* len) {
  size_t L = (n + 255) & ~(size_t)255;
  if (L == 0) L = 256;
  for (auto it = g.va_free.begin(); it != g.va_free.end(); ++it) {
    if (it->second >= L) {
      *off = it->first;
      *len = L;
      size_t rest = it->second - L, start = it->first + L;
      g.va_free.erase(it);
      if (rest) g.va_free[start] = rest;
      return true;
    }
  }
  if (g.va_top + L > g.va_size) return false;
  *off = g.va_top;
  *len = L;
  g.va_top += L;
  return true;
}
void va_release(size_t off, size_t len) {
  auto it = g.va_free.emplace(off, len).first;
  auto next = std::next(it);
  if (next != g.va_free.end() && it->first + it->second == next->first) {
    it->second += next->second;
    g.va_free.erase(next);
  }
  if (it != g.va_free.begin()) {
    auto prev = std::prev(it);
    if (prev->first + prev->second == it->first) {
      prev->second += it->second;
      g.va_free.erase(it);
    }
  }
}

// Find the allocation containing p (one-past-the-end counts, as kernels may receive it).
Alloc* find_alloc(const void* p) {
  uintptr_t a = (uintptr_t)p;
  auto it = g.allocs.upper_bound(a);
  if (it == g.allocs.begin()) return nullptr;
  --it;
  if (a <= it->first + it->second.size) return &it->second;
  return nullptr;
}

const char* cl_err(cl_int e) {
  switch (e) {
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    default: return "OpenCL error";
  }
}

bool build_program(TuInfo* t, std::string& why) {
  cl_int err;
  const char* src = t->cl_src;
  size_t len = strlen(src);
  Program* P = new Program();
  P->prog = p_clCreateProgramWithSource(g.ctx, 1, &src, &len, &err);
  if (err != CL_SUCCESS) {
    why = "clCreateProgramWithSource failed";
    return false;
  }
  std::string opts = g.std_opt;
  if (const char* e = getenv("PK_OPENCL_BUILD_OPTIONS")) opts += std::string(" ") + e;
  err = p_clBuildProgram(P->prog, 1, &g.dev, opts.c_str(), nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t n = 0;
    p_clGetProgramBuildInfo(P->prog, g.dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
    std::string log(n, '\0');
    if (n) p_clGetProgramBuildInfo(P->prog, g.dev, CL_PROGRAM_BUILD_LOG, n, &log[0], nullptr);
    why = std::string(t->file ? t->file : "?") + ": OpenCL build failed on " + g.dev_name + " (" + g.std_opt + ")";
    if (verbose()) logf("OpenCL build log for %s:\n%s", t->file ? t->file : "?", log.c_str());
    return false;
  }
  for (int i = 0; i < t->nkernels; ++i) {
    const KernelMeta& k = t->kernels[i];
    std::vector<std::pair<char, int>> sig;
    const char* s = k.sig;
    while (s && *s) {
      char kind = *s++;
      int size = 0;
      while (*s >= '0' && *s <= '9') size = size * 10 + (*s++ - '0');
      sig.push_back({kind, size});
      if (*s == ',') ++s;
    }
    P->sigs[k.name] = sig;
    std::vector<std::string>& ks = P->ksyms[k.name];
    for (const char* q = k.syms; q && *q;) {
      const char* e = strchr(q, ',');
      ks.emplace_back(q, e ? (size_t)(e - q) : strlen(q));
      q = e ? e + 1 : q + strlen(q);
    }
  }
  t->backend_data = P;
  return true;
}

cl_kernel get_kernel(Program* P, const std::string& name, cl_int* err) {
  auto it = P->kernels.find(name);
  if (it != P->kernels.end()) {
    *err = CL_SUCCESS;
    return it->second;
  }
  cl_kernel k = p_clCreateKernel(P->prog, name.c_str(), err);
  if (*err == CL_SUCCESS) P->kernels[name] = k;
  return k;
}

// Upload managed buffers that the host may have written; mark them device-newer.
cl_int prepare_managed(Alloc* a) {
  if (!a->managed) return CL_SUCCESS;
  cl_int e = CL_SUCCESS;
  if (!a->dev_newer && a->size)
    e = p_clEnqueueWriteBuffer(g.q, a->mem, CL_TRUE, 0, a->size, (void*)a->base, 0, nullptr, nullptr);
  a->dev_newer = true;
  g.pending_managed.insert(a->base);
  return e;
}

cl_int sync_locked() {
  cl_int e = p_clFinish(g.q);
  for (uintptr_t base : g.pending_managed) {
    auto it = g.allocs.find(base);
    if (it == g.allocs.end()) continue;
    Alloc& a = it->second;
    if (a.dev_newer && a.size)
      e |= p_clEnqueueReadBuffer(g.q, a.mem, CL_TRUE, 0, a.size, (void*)a.base, 0, nullptr, nullptr);
    a.dev_newer = false;
  }
  g.pending_managed.clear();
  return e;
}

}  // namespace

bool init(bool allow_cpu, std::string& why) {
  if (!load_library(why)) return false;
  cl_uint np = 0;
  if (p_clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) {
    why = "no OpenCL platform";
    return false;
  }
  std::vector<cl_platform_id> plats(np);
  p_clGetPlatformIDs(np, plats.data(), nullptr);
  struct Cand {
    cl_platform_id p;
    cl_device_id d;
    cl_device_type t;
  };
  std::vector<Cand> devs;
  for (cl_platform_id p : plats) {
    cl_uint nd = 0;
    if (p_clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
    std::vector<cl_device_id> ds(nd);
    p_clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, nd, ds.data(), nullptr);
    for (cl_device_id d : ds) {
      cl_device_type t = 0;
      p_clGetDeviceInfo(d, CL_DEVICE_TYPE, sizeof t, &t, nullptr);
      devs.push_back({p, d, t});
    }
  }
  if (devs.empty()) {
    why = "no OpenCL device";
    return false;
  }
  int pick = -1;
  if (const char* e = getenv("PK_OPENCL_DEVICE")) {
    int i = atoi(e);
    if (i >= 0 && i < (int)devs.size()) pick = i;
  }
  for (int pass = 0; pass < 3 && pick < 0; ++pass) {
    cl_device_type want = pass == 0 ? CL_DEVICE_TYPE_GPU : (pass == 1 ? CL_DEVICE_TYPE_ACCELERATOR : CL_DEVICE_TYPE_CPU);
    if (pass == 2 && !allow_cpu) break;
    for (size_t i = 0; i < devs.size(); ++i)
      if (devs[i].t & want) { pick = (int)i; break; }
  }
  if (pick < 0) {
    why = "only OpenCL CPU devices found (the native CPU back end is used instead; PK_BACKEND=opencl forces them)";
    return false;
  }
  g.plat = devs[pick].p;
  g.dev = devs[pick].d;
  g.dev_name = cl_str_dev(g.dev, CL_DEVICE_NAME);
  p_clGetDeviceInfo(g.dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof g.max_wg, &g.max_wg, nullptr);
  p_clGetDeviceInfo(g.dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof g.max_item, g.max_item, nullptr);
  p_clGetDeviceInfo(g.dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof g.global_mem, &g.global_mem, nullptr);
  p_clGetDeviceInfo(g.dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof g.local_mem, &g.local_mem, nullptr);
  p_clGetDeviceInfo(g.dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof g.max_alloc, &g.max_alloc, nullptr);
  p_clGetDeviceInfo(g.dev, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof g.const_mem, &g.const_mem, nullptr);
  p_clGetDeviceInfo(g.dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof g.cus, &g.cus, nullptr);
  p_clGetDeviceInfo(g.dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof g.clock_mhz, &g.clock_mhz, nullptr);
  p_clGetDeviceInfo(g.dev, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof g.unified, &g.unified, nullptr);

  // Pick the OpenCL C dialect: 3.0 with the generic address space, else 2.0, else 1.2.
  std::string ver = cl_str_dev(g.dev, CL_DEVICE_VERSION);   // "OpenCL X.Y ..."
  int major = 1, minor = 2;
  sscanf(ver.c_str(), "OpenCL %d.%d", &major, &minor);
  std::string cver = cl_str_dev(g.dev, CL_DEVICE_OPENCL_C_VERSION);   // "OpenCL C X.Y ..."
  int cmaj = 1, cmin = 2;
  sscanf(cver.c_str(), "OpenCL C %d.%d", &cmaj, &cmin);
  bool generic = false;
  if (major >= 3) {
    size_t n = 0;
    if (p_clGetDeviceInfo(g.dev, CL_DEVICE_OPENCL_C_FEATURES, 0, nullptr, &n) == CL_SUCCESS && n) {
      std::vector<cl_name_version> f(n / sizeof(cl_name_version));
      p_clGetDeviceInfo(g.dev, CL_DEVICE_OPENCL_C_FEATURES, n, f.data(), nullptr);
      for (auto& x : f)
        if (strcmp(x.name, "__opencl_c_generic_address_space") == 0) generic = true;
    }
  }
  if (major >= 3 && generic)
    g.std_opt = "-cl-std=CL3.0";
  else if (cmaj >= 2)
    g.std_opt = "-cl-std=CL2.0";
  else
    g.std_opt = "-cl-std=CL1.2";
  if (const char* e = getenv("PK_OPENCL_STD")) if (*e) g.std_opt = std::string("-cl-std=") + e;   // e.g. CL1.2 for testing

  cl_int err;
  g.ctx = p_clCreateContext(nullptr, 1, &g.dev, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    why = "clCreateContext failed";
    return false;
  }
  g.q = p_clCreateCommandQueue(g.ctx, g.dev, 0, &err);
  if (err != CL_SUCCESS) {
    why = "clCreateCommandQueue failed";
    return false;
  }
  if (!reserve_address_range()) {
    why = "cannot reserve a device address range";
    return false;
  }
  // Build every translated program now, so an unsupported construct makes the
  // whole program fall back to the CPU before any memory is allocated.
  for (TuInfo* t = tu_list(); t; t = t->next) {
    if (!t->cl_src || !t->cl_ok) continue;
    if (!build_program(t, why)) return false;
  }
  // __device__ / __constant__ variables: one buffer each, initialised from the host copy
  // (which holds the variable's initial value), passed to kernels as hidden arguments.
  for (TuInfo* t = tu_list(); t; t = t->next) {
    Program* P = static_cast<Program*>(t->backend_data);
    if (!P) continue;
    for (int i = 0; i < t->nsymbols; ++i) {
      const SymbolMeta& sm = t->symbols[i];
      void* d = nullptr;
      if (malloc(&d, sm.size, false) != cudaSuccess || memcpy(d, sm.host_addr, sm.size) != cudaSuccess) {
        why = std::string("cannot allocate device variable ") + sm.name;
        return false;
      }
      P->symaddr[sm.name] = d;
    }
  }
  return true;
}

const char* std_option() { return g.std_opt.c_str(); }

void fill_props(cudaDeviceProp* p) {
  std::memset(p, 0, sizeof *p);
  snprintf(p->name, sizeof p->name, "%s (OpenCL via PolyKernel)", g.dev_name.c_str());
  p->totalGlobalMem = (size_t)g.global_mem;
  p->sharedMemPerBlock = (size_t)g.local_mem;
  p->sharedMemPerBlockOptin = (size_t)g.local_mem;
  p->sharedMemPerMultiprocessor = (size_t)g.local_mem;
  p->regsPerBlock = 65536;
  p->regsPerMultiprocessor = 65536;
  p->warpSize = 32;
  p->memPitch = (size_t)g.max_alloc;
  p->maxThreadsPerBlock = (int)g.max_wg;
  for (int i = 0; i < 3; ++i) p->maxThreadsDim[i] = (int)(g.max_item[i] > 0x7fffffff ? 0x7fffffff : g.max_item[i]);
  p->maxGridSize[0] = 0x7fffffff;
  p->maxGridSize[1] = 65535;
  p->maxGridSize[2] = 65535;
  p->clockRate = (int)g.clock_mhz * 1000;
  p->memoryClockRate = p->clockRate;
  p->memoryBusWidth = 128;
  p->totalConstMem = (size_t)g.const_mem;
  p->major = 8;
  p->minor = 0;
  p->textureAlignment = 512;
  p->multiProcessorCount = (int)g.cus;
  p->maxThreadsPerMultiProcessor = (int)g.max_wg * 2;
  p->maxBlocksPerMultiProcessor = 16;
  p->integrated = g.unified ? 1 : 0;
  p->unifiedAddressing = 1;
  p->managedMemory = 1;
  p->computeMode = cudaComputeModeDefault;
  p->l2CacheSize = 1 << 20;
}

cudaError_t malloc(void** out, size_t n, bool managed) {
  std::lock_guard<std::mutex> l(g.m);
  if (g.max_alloc && n > g.max_alloc) return cudaErrorMemoryAllocation;
  cl_int err;
  cl_mem mem = p_clCreateBuffer(g.ctx, CL_MEM_READ_WRITE, n ? n : 1, nullptr, &err);
  if (err != CL_SUCCESS) return cudaErrorMemoryAllocation;
  Alloc a{};
  a.size = n;
  a.mem = mem;
  a.managed = managed;
  if (managed) {
    void* h = nullptr;
#if defined(_WIN32)
    h = _aligned_malloc(n ? n : 1, 256);
#else
    if (posix_memalign(&h, 256, n ? n : 1) != 0) h = nullptr;
#endif
    if (!h) {
      p_clReleaseMemObject(mem);
      return cudaErrorMemoryAllocation;
    }
    std::memset(h, 0, n);
    a.base = (uintptr_t)h;
  } else {
    if (!va_alloc(n, &a.va_off, &a.va_len)) {
      p_clReleaseMemObject(mem);
      return cudaErrorMemoryAllocation;
    }
    a.base = g.va_base + a.va_off;
  }
  g.allocs[a.base] = a;
  *out = (void*)a.base;
  return cudaSuccess;
}

cudaError_t free(void* p) {
  std::lock_guard<std::mutex> l(g.m);
  auto it = g.allocs.find((uintptr_t)p);
  if (it == g.allocs.end()) return cudaErrorInvalidDevicePointer;
  Alloc a = it->second;
  g.allocs.erase(it);
  g.pending_managed.erase(a.base);
  p_clReleaseMemObject(a.mem);
  if (a.managed) {
#if defined(_WIN32)
    _aligned_free((void*)a.base);
#else
    ::free((void*)a.base);
#endif
  } else {
    va_release(a.va_off, a.va_len);
  }
  return cudaSuccess;
}

cudaError_t memcpy(void* dst, const void* src, size_t n) {
  std::lock_guard<std::mutex> l(g.m);
  sync_locked();
  Alloc* d = find_alloc(dst);
  Alloc* s = find_alloc(src);
  if (d && d->managed) d = nullptr;   // managed memory is host memory between launches
  if (s && s->managed) s = nullptr;
  cl_int e = CL_SUCCESS;
  if (d && (uintptr_t)dst + n > d->base + d->size) return cudaErrorInvalidValue;
  if (s && (uintptr_t)src + n > s->base + s->size) return cudaErrorInvalidValue;
  if (d && s) {
    e = p_clEnqueueCopyBuffer(g.q, s->mem, d->mem, (uintptr_t)src - s->base, (uintptr_t)dst - d->base, n, 0, nullptr,
                              nullptr);
    if (e == CL_SUCCESS) e = p_clFinish(g.q);
  } else if (d) {
    e = p_clEnqueueWriteBuffer(g.q, d->mem, CL_TRUE, (uintptr_t)dst - d->base, n, src, 0, nullptr, nullptr);
  } else if (s) {
    e = p_clEnqueueReadBuffer(g.q, s->mem, CL_TRUE, (uintptr_t)src - s->base, n, dst, 0, nullptr, nullptr);
  } else {
    std::memmove(dst, src, n);
  }
  return e == CL_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t memset(void* p, int v, size_t n) {
  std::lock_guard<std::mutex> l(g.m);
  Alloc* a = find_alloc(p);
  if (!a) return cudaErrorInvalidDevicePointer;
  if ((uintptr_t)p + n > a->base + a->size) return cudaErrorInvalidValue;
  if (a->managed) {
    sync_locked();
    std::memset(p, v, n);
    return cudaSuccess;
  }
  cl_uchar pat = (cl_uchar)v;
  cl_int e = p_clEnqueueFillBuffer(g.q, a->mem, &pat, 1, (uintptr_t)p - a->base, n, 0, nullptr, nullptr);
  return e == CL_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t synchronize() {
  std::lock_guard<std::mutex> l(g.m);
  return sync_locked() == CL_SUCCESS ? cudaSuccess : cudaErrorLaunchFailure;
}

int pointer_type(const void* p) {
  std::lock_guard<std::mutex> l(g.m);
  Alloc* a = find_alloc(p);
  if (!a) return cudaMemoryTypeUnregistered;
  return a->managed ? cudaMemoryTypeManaged : cudaMemoryTypeDevice;
}

void mem_info(size_t* fr, size_t* total) {
  std::lock_guard<std::mutex> l(g.m);
  size_t used = 0;
  for (auto& kv : g.allocs) used += kv.second.size;
  if (total) *total = (size_t)g.global_mem;
  if (fr) *fr = used < g.global_mem ? (size_t)g.global_mem - used : 0;
}

static void* symbol_buffer(const SymbolMeta* s, const TuInfo* tu) {
  Program* P = static_cast<Program*>(tu->backend_data);
  if (!P) return nullptr;
  auto it = P->symaddr.find(s->name);
  return it == P->symaddr.end() ? nullptr : it->second;
}

cudaError_t to_symbol(const SymbolMeta* s, const TuInfo* tu, const void* src, size_t n, size_t off) {
  char* d = static_cast<char*>(symbol_buffer(s, tu));
  if (!d) return cudaErrorInvalidSymbol;
  return n ? memcpy(d + off, src, n) : cudaSuccess;
}
cudaError_t from_symbol(const SymbolMeta* s, const TuInfo* tu, void* dst, size_t n, size_t off) {
  char* d = static_cast<char*>(symbol_buffer(s, tu));
  if (!d) return cudaErrorInvalidSymbol;
  return n ? memcpy(dst, d + off, n) : cudaSuccess;
}
void* symbol_address(const SymbolMeta* s, const TuInfo* tu) { return symbol_buffer(s, tu); }

}  // namespace ocl

// ---------------------------------------------------------------------------
// Kernel launch
// ---------------------------------------------------------------------------
cudaError_t opencl_launch(const TuInfo* tu, const char* name, const LaunchCfg& c, const ArgPack& a) {
  using namespace ocl;
  const dim3 gr = c.grid, b = c.block;
  size_t nthreads = (size_t)b.x * b.y * b.z;
  if (nthreads > 1024 || b.z > 64)   // as CUDA 12 reports it
    return record_launch_error(cudaErrorInvalidValue);
  if (gr.x == 0 || gr.y == 0 || gr.z == 0 || b.x == 0 || b.y == 0 || b.z == 0 || nthreads > g.max_wg ||
      b.x > g.max_item[0] || b.y > g.max_item[1] || b.z > g.max_item[2])
    return record_launch_error(cudaErrorInvalidConfiguration);
  if (c.shmem > g.local_mem) return record_launch_error(cudaErrorInvalidValue);
  if (a.overflow) return record_launch_error(cudaErrorInvalidValue);

  std::lock_guard<std::mutex> l(g.m);
  // Locate the kernel: this TU first, then any other TU (kernel defined elsewhere).
  Program* P = nullptr;
  const TuInfo* owner = nullptr;
  for (int pass = 0; pass < 2 && !P; ++pass) {
    for (const TuInfo* t = pass == 0 ? tu : tu_list(); t; t = (pass == 0 ? nullptr : t->next)) {
      Program* cand = static_cast<Program*>(t->backend_data);
      if (cand && cand->sigs.count(name)) {
        P = cand;
        owner = t;
        break;
      }
    }
  }
  (void)owner;
  if (!P) {
    logf("kernel '%s' has no OpenCL translation", name);
    return record_launch_error(cudaErrorInvalidDeviceFunction);
  }
  cl_int err;
  cl_kernel k = get_kernel(P, name, &err);
  if (err != CL_SUCCESS) {
    logf("clCreateKernel(%s) failed: %s", name, cl_err(err));
    return record_launch_error(cudaErrorInvalidDeviceFunction);
  }
  const std::vector<std::pair<char, int>>& sig = P->sigs[name];
  if ((int)sig.size() != a.n) {
    logf("kernel '%s': argument count mismatch (%d vs %d)", name, a.n, (int)sig.size());
    return record_launch_error(cudaErrorInvalidValue);
  }
  cl_uint ai = 0;
  for (int i = 0; i < a.n; ++i) {
    const ArgPack::Arg& x = a.args[i];
    char kind = sig[i].first;
    int size = sig[i].second;
    if (kind == 'P') {
      cl_mem mem = nullptr;
      cl_ulong off = 0;
      if (x.kind != ArgPack::Ptr && !(x.kind == ArgPack::Int && x.bits == 0)) {
        logf("kernel '%s': argument %d must be a pointer", name, i);
        return record_launch_error(cudaErrorInvalidValue);
      }
      if (x.kind == ArgPack::Ptr && x.data) {
        Alloc* al = find_alloc(x.data);
        if (!al) {
          logf("kernel '%s': argument %d (%p) is not a device pointer (use cudaMalloc/cudaMallocManaged)", name, i,
               x.data);
          return record_launch_error(cudaErrorInvalidDevicePointer);
        }
        if (al->managed && prepare_managed(al) != CL_SUCCESS) return record_launch_error(cudaErrorLaunchFailure);
        mem = al->mem;
        off = (uintptr_t)x.data - al->base;
      }
      err = p_clSetKernelArg(k, ai++, sizeof(cl_mem), mem ? &mem : nullptr);
      if (err == CL_SUCCESS) err = p_clSetKernelArg(k, ai++, sizeof(cl_ulong), &off);
    } else if (kind == 'I') {
      uint64_t v = x.bits;
      if (x.kind == ArgPack::Flt || x.kind == ArgPack::Dbl) {
        double d;
        std::memcpy(&d, &x.bits, sizeof d);
        v = (uint64_t)(int64_t)d;
      }
      unsigned char buf[8];
      if (size == 1) { uint8_t t = (uint8_t)v; std::memcpy(buf, &t, 1); }
      else if (size == 2) { uint16_t t = (uint16_t)v; std::memcpy(buf, &t, 2); }
      else if (size == 4) { uint32_t t = (uint32_t)v; std::memcpy(buf, &t, 4); }
      else { size = 8; std::memcpy(buf, &v, 8); }
      err = p_clSetKernelArg(k, ai++, (size_t)size, buf);
    } else if (kind == 'F') {
      double d;
      if (x.kind == ArgPack::Int)
        d = x.is_signed ? (double)(int64_t)x.bits : (double)x.bits;
      else
        std::memcpy(&d, &x.bits, sizeof d);
      if (size == 4) {
        float f = (float)d;
        err = p_clSetKernelArg(k, ai++, 4, &f);
      } else {
        err = p_clSetKernelArg(k, ai++, 8, &d);
      }
    } else {   // 'S': raw bytes, layout shared between host C++ and OpenCL C
      if (x.kind == ArgPack::Raw)
        err = p_clSetKernelArg(k, ai++, x.size, x.data);
      else
        err = p_clSetKernelArg(k, ai++, x.size, &x.bits);
    }
    if (err != CL_SUCCESS) {
      logf("kernel '%s': setting argument %d failed (%s)", name, i, cl_err(err));
      return record_launch_error(cudaErrorInvalidValue);
    }
  }
  for (const std::string& sn : P->ksyms[name]) {   // __device__/__constant__ variables used by the kernel
    auto it = P->symaddr.find(sn);
    Alloc* al = it == P->symaddr.end() ? nullptr : find_alloc(it->second);
    if (!al) {
      logf("kernel '%s': device variable '%s' has no buffer", name, sn.c_str());
      return record_launch_error(cudaErrorInvalidSymbol);
    }
    err = p_clSetKernelArg(k, ai++, sizeof(cl_mem), &al->mem);
    if (err != CL_SUCCESS) {
      logf("kernel '%s': setting device variable '%s' failed (%s)", name, sn.c_str(), cl_err(err));
      return record_launch_error(cudaErrorInvalidValue);
    }
  }
  size_t dyn = c.shmem < 16 ? 16 : c.shmem;
  err = p_clSetKernelArg(k, ai++, dyn, nullptr);   // dynamic __shared__ memory
  if (err != CL_SUCCESS) return record_launch_error(cudaErrorInvalidValue);
  size_t global[3] = {(size_t)gr.x * b.x, (size_t)gr.y * b.y, (size_t)gr.z * b.z};
  size_t local[3] = {b.x, b.y, b.z};
  err = p_clEnqueueNDRangeKernel(g.q, k, 3, nullptr, global, local, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    logf("kernel '%s': launch failed (%s)", name, cl_err(err));
    return record_launch_error(err == CL_INVALID_WORK_GROUP_SIZE ? cudaErrorInvalidConfiguration
                                                                 : cudaErrorLaunchOutOfResources);
  }
  static const bool sync_each = env_size("PK_SYNC_LAUNCH", 0) != 0;
  if (sync_each) p_clFinish(g.q);
  return cudaSuccess;
}

}  // namespace pk
