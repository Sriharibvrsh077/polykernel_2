// PolyKernel: CUDA runtime API declarations (C linkage, as in the public API).
#ifndef POLYKERNEL_PK_API_H
#define POLYKERNEL_PK_API_H

#include "pk_types.h"

extern "C" {

// Errors
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorString(cudaError_t error);
const char* cudaGetErrorName(cudaError_t error);

// Device management
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags);
cudaError_t cudaChooseDevice(int* device, const struct cudaDeviceProp* prop);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device);
cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaThreadSynchronize(void);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaThreadExit(void);
cudaError_t cudaSetDeviceFlags(unsigned int flags);
cudaError_t cudaGetDeviceFlags(unsigned int* flags);
cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig);
cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache* pCacheConfig);
cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config);
cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig* pConfig);
cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);
cudaError_t cudaDeviceGetLimit(size_t* pValue, enum cudaLimit limit);
cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice);
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device);
cudaError_t cudaDeviceGetP2PAttribute(int* value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);
cudaError_t cudaCtxResetPersistingL2Cache(void);
cudaError_t cudaStreamSetAttribute(cudaStream_t stream, enum cudaLaunchAttributeID attr,
                                   const union cudaLaunchAttributeValue* value);
cudaError_t cudaStreamGetAttribute(cudaStream_t stream, enum cudaLaunchAttributeID attr,
                                   union cudaLaunchAttributeValue* value);
cudaError_t cudaDriverGetVersion(int* driverVersion);
cudaError_t cudaRuntimeGetVersion(int* runtimeVersion);
cudaError_t cudaMemGetInfo(size_t* free, size_t* total);

// Memory
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags);
cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost);
cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void* ptr);
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal);
cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height);
cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream);
cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind,
                            cudaStream_t stream = 0);
cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                         size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                              size_t height, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count);
cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count,
                                cudaStream_t stream = 0);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream = 0);
cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height);
cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height,
                              cudaStream_t stream = 0);
cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream = 0);
cudaError_t cudaMemAdvise(const void* devPtr, size_t count, enum cudaMemoryAdvise advice, int device);
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes* attributes, const void* ptr);

// Symbols (the C++ template overloads below take the variable itself)
cudaError_t pkMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset,
                             enum cudaMemcpyKind kind);
cudaError_t pkMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset,
                               enum cudaMemcpyKind kind);
cudaError_t pkGetSymbolAddress(void** devPtr, const void* symbol);
cudaError_t pkGetSymbolSize(size_t* size, const void* symbol);

// Streams
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0);
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData,
                                  unsigned int flags);
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData);

// Events
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaEventDestroy(cudaEvent_t event);

// Execution control
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                             size_t sharedMem = 0, cudaStream_t stream = 0);
cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache cacheConfig);
cudaError_t cudaFuncSetAttribute(const void* func, enum cudaFuncAttribute attr, int value);
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, const void* func);
cudaError_t cudaFuncSetSharedMemConfig(const void* func, enum cudaSharedMemConfig config);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize,
                                                          size_t dynamicSMemSize);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func,
                                                                   int blockSize, size_t dynamicSMemSize,
                                                                   unsigned int flags);

// Profiler (cuda_profiler_api.h)
cudaError_t cudaProfilerStart(void);
cudaError_t cudaProfilerStop(void);

}  // extern "C"

// ---------------------------------------------------------------------------
// C++ convenience overloads (mirroring cuda_runtime.h)
// ---------------------------------------------------------------------------
template <class T>
static inline cudaError_t cudaMalloc(T** devPtr, size_t size) {
  return ::cudaMalloc((void**)(void*)devPtr, size);
}
template <class T>
static inline cudaError_t cudaMallocHost(T** ptr, size_t size) {
  return ::cudaMallocHost((void**)(void*)ptr, size);
}
template <class T>
static inline cudaError_t cudaMallocHost(T** ptr, size_t size, unsigned int flags) {
  return ::cudaHostAlloc((void**)(void*)ptr, size, flags);
}
template <class T>
static inline cudaError_t cudaHostAlloc(T** ptr, size_t size, unsigned int flags) {
  return ::cudaHostAlloc((void**)(void*)ptr, size, flags);
}
template <class T>
static inline cudaError_t cudaHostGetDevicePointer(T** pDevice, void* pHost, unsigned int flags) {
  return ::cudaHostGetDevicePointer((void**)(void*)pDevice, pHost, flags);
}
template <class T>
static inline cudaError_t cudaMallocManaged(T** devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal) {
  return ::cudaMallocManaged((void**)(void*)devPtr, size, flags);
}
template <class T>
static inline cudaError_t cudaMallocPitch(T** devPtr, size_t* pitch, size_t width, size_t height) {
  return ::cudaMallocPitch((void**)(void*)devPtr, pitch, width, height);
}
template <class T>
static inline cudaError_t cudaMallocAsync(T** devPtr, size_t size, cudaStream_t stream) {
  return ::cudaMallocAsync((void**)(void*)devPtr, size, stream);
}
template <class T>
static inline cudaError_t cudaMemcpyToSymbol(const T& symbol, const void* src, size_t count, size_t offset = 0,
                                             enum cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
  return pkMemcpyToSymbol((const void*)std::addressof(symbol), src, count, offset, kind);
}
template <class T>
static inline cudaError_t cudaMemcpyToSymbolAsync(const T& symbol, const void* src, size_t count,
                                                  size_t offset = 0,
                                                  enum cudaMemcpyKind kind = cudaMemcpyHostToDevice,
                                                  cudaStream_t stream = 0) {
  (void)stream;
  return pkMemcpyToSymbol((const void*)std::addressof(symbol), src, count, offset, kind);
}
template <class T>
static inline cudaError_t cudaMemcpyFromSymbol(void* dst, const T& symbol, size_t count, size_t offset = 0,
                                               enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost) {
  return pkMemcpyFromSymbol(dst, (const void*)std::addressof(symbol), count, offset, kind);
}
template <class T>
static inline cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const T& symbol, size_t count, size_t offset = 0,
                                                    enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost,
                                                    cudaStream_t stream = 0) {
  (void)stream;
  return pkMemcpyFromSymbol(dst, (const void*)std::addressof(symbol), count, offset, kind);
}
template <class T>
static inline cudaError_t cudaGetSymbolAddress(void** devPtr, const T& symbol) {
  return pkGetSymbolAddress(devPtr, (const void*)std::addressof(symbol));
}
template <class T>
static inline cudaError_t cudaGetSymbolSize(size_t* size, const T& symbol) {
  (void)symbol;   // the size is known here, whichever back end holds the variable
  if (!size) return cudaErrorInvalidValue;
  *size = sizeof(T);
  return cudaSuccess;
}
static inline cudaError_t cudaEventCreate(cudaEvent_t* event, unsigned int flags) {
  return ::cudaEventCreateWithFlags(event, flags);
}
template <class T>
static inline cudaError_t cudaFuncSetCacheConfig(T* func, enum cudaFuncCache cacheConfig) {
  return ::cudaFuncSetCacheConfig((const void*)func, cacheConfig);
}
template <class T>
static inline cudaError_t cudaFuncSetAttribute(T* func, enum cudaFuncAttribute attr, int value) {
  return ::cudaFuncSetAttribute((const void*)func, attr, value);
}
template <class T>
static inline cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, T* func) {
  return ::cudaFuncGetAttributes(attr, (const void*)func);
}
template <class T>
static inline cudaError_t cudaLaunchKernel(T* func, dim3 gridDim, dim3 blockDim, void** args,
                                           size_t sharedMem = 0, cudaStream_t stream = 0) {
  return ::cudaLaunchKernel((const void*)func, gridDim, blockDim, args, sharedMem, stream);
}
template <class T>
static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, T func, int blockSize,
                                                                        size_t dynamicSMemSize) {
  return ::cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, (const void*)func, blockSize,
                                                         dynamicSMemSize);
}
template <class T>
static inline cudaError_t cudaOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, T func,
                                                             size_t dynamicSMemSize = 0,
                                                             int blockSizeLimit = 0) {
  (void)func;
  (void)dynamicSMemSize;
  int bs = 256;
  if (blockSizeLimit > 0 && blockSizeLimit < bs) bs = blockSizeLimit;
  int dev = 0, sms = 1;
  ::cudaGetDevice(&dev);
  ::cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (blockSize) *blockSize = bs;
  if (minGridSize) *minGridSize = 4 * (sms > 0 ? sms : 1);
  return cudaSuccess;
}
template <class T, class UnaryFunction>
static inline cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMem(int* minGridSize, int* blockSize,
                                                                         T func, UnaryFunction smemFn,
                                                                         int blockSizeLimit = 0) {
  (void)smemFn;
  return cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, 0, blockSizeLimit);
}

#endif  // POLYKERNEL_PK_API_H
