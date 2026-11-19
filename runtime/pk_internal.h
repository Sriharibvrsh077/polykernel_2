// PolyKernel runtime internals (not installed for user code).
#ifndef POLYKERNEL_PK_INTERNAL_H
#define POLYKERNEL_PK_INTERNAL_H

#include "cuda_runtime.h"
#include <string>

namespace pk {

enum class Backend { Cpu, OpenCL, Ptx };
Backend backend();
cudaError_t set_error(cudaError_t e);   // records e as the last error when it is not cudaSuccess
bool verbose();
void logf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
TuInfo* tu_list();
const char* arch_name();
size_t env_size(const char* name, size_t def);

namespace cpu {
int num_threads();
}

namespace ocl {
bool init(bool allow_cpu_devices, std::string& why);
void fill_props(cudaDeviceProp* p);
const char* std_option();   // "-cl-std=CLx.y" chosen for this device
int attribute(cudaDeviceAttr a, int* ok);
cudaError_t malloc(void** p, size_t n, bool managed);
cudaError_t free(void* p);
cudaError_t memcpy(void* dst, const void* src, size_t n);
cudaError_t memset(void* p, int v, size_t n);
cudaError_t synchronize();
cudaError_t to_symbol(const SymbolMeta* s, const TuInfo* tu, const void* src, size_t n, size_t off);
cudaError_t from_symbol(const SymbolMeta* s, const TuInfo* tu, void* dst, size_t n, size_t off);
void* symbol_address(const SymbolMeta* s, const TuInfo* tu);   // device address of a __device__ variable
int pointer_type(const void* p);   // 0 unknown, 2 device, 3 managed
void mem_info(size_t* free_b, size_t* total_b);
}  // namespace ocl

namespace ptx {   // NVIDIA GPUs through the driver API, running PTX from PolyKernel's own compiler
bool init(std::string& why);
const char* device_name();
int driver_version();
void fill_props(cudaDeviceProp* p);
int attribute(cudaDeviceAttr a, int* ok);
cudaError_t malloc(void** p, size_t n, bool managed);
cudaError_t host_alloc(void** p, size_t n);
cudaError_t host_device_pointer(void** d, void* h);
cudaError_t free(void* p);
cudaError_t memcpy(void* dst, const void* src, size_t n);
cudaError_t memset(void* p, int v, size_t n);
cudaError_t synchronize();
cudaError_t to_symbol(const SymbolMeta* s, const TuInfo* tu, const void* src, size_t n, size_t off);
cudaError_t from_symbol(const SymbolMeta* s, const TuInfo* tu, void* dst, size_t n, size_t off);
void* symbol_address(const SymbolMeta* s, const TuInfo* tu);
int pointer_type(const void* p);
void mem_info(size_t* free_b, size_t* total_b);
cudaError_t launch(const TuInfo* tu, const char* name, const LaunchCfg& c, const ArgPack& a);
}  // namespace ptx

// The active GPU back end (PTX or OpenCL), whichever was chosen at start-up.
bool on_gpu();
namespace dev {
#define PK_DEV_FN(ret, name, params, args) \
  inline ret name params { return backend() == Backend::Ptx ? ptx::name args : ocl::name args; }
PK_DEV_FN(void, fill_props, (cudaDeviceProp* p), (p))
PK_DEV_FN(int, attribute, (cudaDeviceAttr a, int* ok), (a, ok))
PK_DEV_FN(cudaError_t, malloc, (void** p, size_t n, bool managed), (p, n, managed))
PK_DEV_FN(cudaError_t, free, (void* p), (p))
PK_DEV_FN(cudaError_t, memcpy, (void* dst, const void* src, size_t n), (dst, src, n))
PK_DEV_FN(cudaError_t, memset, (void* p, int v, size_t n), (p, v, n))
PK_DEV_FN(cudaError_t, synchronize, (), ())
PK_DEV_FN(cudaError_t, to_symbol, (const SymbolMeta* s, const TuInfo* tu, const void* src, size_t n, size_t off),
          (s, tu, src, n, off))
PK_DEV_FN(cudaError_t, from_symbol, (const SymbolMeta* s, const TuInfo* tu, void* dst, size_t n, size_t off),
          (s, tu, dst, n, off))
PK_DEV_FN(void*, symbol_address, (const SymbolMeta* s, const TuInfo* tu), (s, tu))
PK_DEV_FN(int, pointer_type, (const void* p), (p))
PK_DEV_FN(void, mem_info, (size_t* f, size_t* t), (f, t))
#undef PK_DEV_FN
}  // namespace dev

}  // namespace pk

#endif
