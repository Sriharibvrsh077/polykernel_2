// PolyKernel - a clean-room, source-compatible implementation of the CUDA runtime API.
// Written from the public CUDA programming guide / runtime API reference only.
// No NVIDIA headers, libraries or binaries are used or reverse engineered.
//
// pk_types.h: built-in vector types, dim3, enums and structs of the runtime API.
#ifndef POLYKERNEL_PK_TYPES_H
#define POLYKERNEL_PK_TYPES_H

#ifndef __cplusplus
#error "PolyKernel's CUDA headers require C++ (compile .cu files with pkcc)"
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cfloat>
#include <new>
#include <type_traits>
#include <utility>
#include <tuple>

#define POLYKERNEL 1
// Include-guard names of the corresponding NVIDIA headers: existing code (for example
// the samples' helper_cuda.h) tests these to detect that the runtime API is available.
#define __DRIVER_TYPES_H__
#define __VECTOR_TYPES_H__
#define __CUDA_RUNTIME_API_H__
#define __HOST_DEFINES_H__
#define __VECTOR_FUNCTIONS_H__
#define POLYKERNEL_VERSION_MAJOR 0
#define POLYKERNEL_VERSION_MINOR 1
#ifndef CUDART_VERSION
#define CUDART_VERSION 12000
#endif
#ifndef CUDA_VERSION
#define CUDA_VERSION 12000
#endif
#define __CUDART_API_VERSION CUDART_VERSION

// ---------------------------------------------------------------------------
// Execution-space and memory-space qualifiers.
// In a .cu file compiled by pkcc these tokens are rewritten by the translator
// (pkcc defines PK_TRANSLATING while preprocessing so they stay visible).
// The fallbacks below let plain C++ files that include cuda_runtime.h build.
// ---------------------------------------------------------------------------
#ifndef PK_TRANSLATING
#define __host__
#define __device__
#define __global__
#define __shared__ static thread_local
#define __constant__
#define __managed__
#endif
#define __forceinline__ inline __attribute__((always_inline))
// __noinline__ is rewritten by pkcc (libstdc++ uses the same token inside __attribute__)
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(...)
#define __grid_constant__
#define __cudart_builtin__
#define __device_builtin__
#define CUDARTAPI
#define CUDART_CB

// ---------------------------------------------------------------------------
// Built-in vector types (sizes and alignments follow the CUDA programming guide)
// ---------------------------------------------------------------------------
#define PK_VEC_TYPES(BASE, T, A1, A2, A3, A4)                                              \
  struct __attribute__((aligned(A1))) BASE##1 { T x; };                                     \
  struct __attribute__((aligned(A2))) BASE##2 { T x, y; };                                  \
  struct __attribute__((aligned(A3))) BASE##3 { T x, y, z; };                               \
  struct __attribute__((aligned(A4))) BASE##4 { T x, y, z, w; };                            \
  static inline BASE##1 make_##BASE##1(T x) { BASE##1 r; r.x = x; return r; }               \
  static inline BASE##2 make_##BASE##2(T x, T y) { BASE##2 r; r.x = x; r.y = y; return r; } \
  static inline BASE##3 make_##BASE##3(T x, T y, T z) {                                     \
    BASE##3 r; r.x = x; r.y = y; r.z = z; return r;                                         \
  }                                                                                         \
  static inline BASE##4 make_##BASE##4(T x, T y, T z, T w) {                                \
    BASE##4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r;                                \
  }

PK_VEC_TYPES(char, signed char, 1, 2, 1, 4)
PK_VEC_TYPES(uchar, unsigned char, 1, 2, 1, 4)
PK_VEC_TYPES(short, short, 2, 4, 2, 8)
PK_VEC_TYPES(ushort, unsigned short, 2, 4, 2, 8)
PK_VEC_TYPES(int, int, 4, 8, 4, 16)
PK_VEC_TYPES(uint, unsigned int, 4, 8, 4, 16)
PK_VEC_TYPES(long, long, sizeof(long), 2 * sizeof(long), sizeof(long), 16)
PK_VEC_TYPES(ulong, unsigned long, sizeof(long), 2 * sizeof(long), sizeof(long), 16)
PK_VEC_TYPES(longlong, long long, 8, 16, 8, 16)
PK_VEC_TYPES(ulonglong, unsigned long long, 8, 16, 8, 16)
PK_VEC_TYPES(float, float, 4, 8, 4, 16)
PK_VEC_TYPES(double, double, 8, 16, 8, 16)
#undef PK_VEC_TYPES

struct dim3 {
  unsigned int x, y, z;
  constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
  constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
  constexpr operator uint3() const { return uint3{x, y, z}; }
};

// ---------------------------------------------------------------------------
// Error codes (numeric values from the public runtime API reference)
// ---------------------------------------------------------------------------
enum cudaError {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorCudartUnloading = 4,
  cudaErrorProfilerDisabled = 5,
  cudaErrorInvalidConfiguration = 9,
  cudaErrorInvalidPitchValue = 12,
  cudaErrorInvalidSymbol = 13,
  cudaErrorInvalidHostPointer = 16,
  cudaErrorInvalidDevicePointer = 17,
  cudaErrorInvalidTexture = 18,
  cudaErrorInvalidChannelDescriptor = 20,
  cudaErrorInvalidMemcpyDirection = 21,
  cudaErrorInvalidFilterSetting = 26,
  cudaErrorInvalidNormSetting = 27,
  cudaErrorStubLibrary = 34,
  cudaErrorInsufficientDriver = 35,
  cudaErrorInvalidSurface = 37,
  cudaErrorDuplicateVariableName = 43,
  cudaErrorDevicesUnavailable = 46,
  cudaErrorIncompatibleDriverContext = 49,
  cudaErrorMissingConfiguration = 52,
  cudaErrorLaunchMaxDepthExceeded = 65,
  cudaErrorInvalidDeviceFunction = 98,
  cudaErrorNoDevice = 100,
  cudaErrorInvalidDevice = 101,
  cudaErrorDeviceNotLicensed = 102,
  cudaErrorStartupFailure = 127,
  cudaErrorInvalidKernelImage = 200,
  cudaErrorDeviceUninitialized = 201,
  cudaErrorMapBufferObjectFailed = 205,
  cudaErrorUnmapBufferObjectFailed = 206,
  cudaErrorNoKernelImageForDevice = 209,
  cudaErrorPeerAccessUnsupported = 217,
  cudaErrorInvalidSource = 300,
  cudaErrorFileNotFound = 301,
  cudaErrorSharedObjectSymbolNotFound = 302,
  cudaErrorSharedObjectInitFailed = 303,
  cudaErrorOperatingSystem = 304,
  cudaErrorInvalidResourceHandle = 400,
  cudaErrorIllegalState = 401,
  cudaErrorSymbolNotFound = 500,
  cudaErrorNotReady = 600,
  cudaErrorIllegalAddress = 700,
  cudaErrorLaunchOutOfResources = 701,
  cudaErrorLaunchTimeout = 702,
  cudaErrorPeerAccessAlreadyEnabled = 704,
  cudaErrorPeerAccessNotEnabled = 705,
  cudaErrorSetOnActiveProcess = 708,
  cudaErrorContextIsDestroyed = 709,
  cudaErrorAssert = 710,
  cudaErrorHostMemoryAlreadyRegistered = 712,
  cudaErrorHostMemoryNotRegistered = 713,
  cudaErrorHardwareStackError = 714,
  cudaErrorIllegalInstruction = 715,
  cudaErrorMisalignedAddress = 716,
  cudaErrorInvalidAddressSpace = 717,
  cudaErrorInvalidPc = 718,
  cudaErrorLaunchFailure = 719,
  cudaErrorCooperativeLaunchTooLarge = 720,
  cudaErrorNotPermitted = 800,
  cudaErrorNotSupported = 801,
  cudaErrorUnknown = 999
};
typedef enum cudaError cudaError_t;

enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};

enum cudaMemoryType {
  cudaMemoryTypeUnregistered = 0,
  cudaMemoryTypeHost = 1,
  cudaMemoryTypeDevice = 2,
  cudaMemoryTypeManaged = 3
};

struct cudaPointerAttributes {
  enum cudaMemoryType type;
  int device;
  void* devicePointer;
  void* hostPointer;
};

enum cudaFuncCache {
  cudaFuncCachePreferNone = 0,
  cudaFuncCachePreferShared = 1,
  cudaFuncCachePreferL1 = 2,
  cudaFuncCachePreferEqual = 3
};

enum cudaSharedMemConfig {
  cudaSharedMemBankSizeDefault = 0,
  cudaSharedMemBankSizeFourByte = 1,
  cudaSharedMemBankSizeEightByte = 2
};

enum cudaFuncAttribute {
  cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
  cudaFuncAttributePreferredSharedMemoryCarveout = 9,
  cudaFuncAttributeClusterDimMustBeSet = 10,
  cudaFuncAttributeRequiredClusterWidth = 11,
  cudaFuncAttributeRequiredClusterHeight = 12,
  cudaFuncAttributeRequiredClusterDepth = 13,
  cudaFuncAttributeNonPortableClusterSizeAllowed = 14,
  cudaFuncAttributeClusterSchedulingPolicyPreference = 15,
  cudaFuncAttributeMax
};

enum cudaLimit {
  cudaLimitStackSize = 0x00,
  cudaLimitPrintfFifoSize = 0x01,
  cudaLimitMallocHeapSize = 0x02,
  cudaLimitDevRuntimeSyncDepth = 0x03,
  cudaLimitDevRuntimePendingLaunchCount = 0x04,
  cudaLimitMaxL2FetchGranularity = 0x05,
  cudaLimitPersistingL2CacheSize = 0x06
};

enum cudaComputeMode {
  cudaComputeModeDefault = 0,
  cudaComputeModeExclusive = 1,
  cudaComputeModeProhibited = 2,
  cudaComputeModeExclusiveProcess = 3
};

enum cudaMemoryAdvise {
  cudaMemAdviseSetReadMostly = 1,
  cudaMemAdviseUnsetReadMostly = 2,
  cudaMemAdviseSetPreferredLocation = 3,
  cudaMemAdviseUnsetPreferredLocation = 4,
  cudaMemAdviseSetAccessedBy = 5,
  cudaMemAdviseUnsetAccessedBy = 6
};

enum cudaDeviceAttr {
  cudaDevAttrMaxThreadsPerBlock = 1,
  cudaDevAttrMaxBlockDimX = 2,
  cudaDevAttrMaxBlockDimY = 3,
  cudaDevAttrMaxBlockDimZ = 4,
  cudaDevAttrMaxGridDimX = 5,
  cudaDevAttrMaxGridDimY = 6,
  cudaDevAttrMaxGridDimZ = 7,
  cudaDevAttrMaxSharedMemoryPerBlock = 8,
  cudaDevAttrTotalConstantMemory = 9,
  cudaDevAttrWarpSize = 10,
  cudaDevAttrMaxPitch = 11,
  cudaDevAttrMaxRegistersPerBlock = 12,
  cudaDevAttrClockRate = 13,
  cudaDevAttrTextureAlignment = 14,
  cudaDevAttrGpuOverlap = 15,
  cudaDevAttrMultiProcessorCount = 16,
  cudaDevAttrKernelExecTimeout = 17,
  cudaDevAttrIntegrated = 18,
  cudaDevAttrCanMapHostMemory = 19,
  cudaDevAttrComputeMode = 20,
  cudaDevAttrConcurrentKernels = 31,
  cudaDevAttrEccEnabled = 32,
  cudaDevAttrPciBusId = 33,
  cudaDevAttrPciDeviceId = 34,
  cudaDevAttrTccDriver = 35,
  cudaDevAttrMemoryClockRate = 36,
  cudaDevAttrGlobalMemoryBusWidth = 37,
  cudaDevAttrL2CacheSize = 38,
  cudaDevAttrMaxThreadsPerMultiProcessor = 39,
  cudaDevAttrAsyncEngineCount = 40,
  cudaDevAttrUnifiedAddressing = 41,
  cudaDevAttrComputeCapabilityMajor = 75,
  cudaDevAttrComputeCapabilityMinor = 76,
  cudaDevAttrStreamPrioritiesSupported = 78,
  cudaDevAttrGlobalL1CacheSupported = 79,
  cudaDevAttrLocalL1CacheSupported = 80,
  cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
  cudaDevAttrMaxRegistersPerMultiprocessor = 82,
  cudaDevAttrManagedMemory = 83,
  cudaDevAttrIsMultiGpuBoard = 84,
  cudaDevAttrPageableMemoryAccess = 88,
  cudaDevAttrConcurrentManagedAccess = 89,
  cudaDevAttrCooperativeLaunch = 95,
  cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
  cudaDevAttrMemoryPoolsSupported = 115
};

struct cudaDeviceProp {
  char name[256];
  char uuid[16];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  size_t texturePitchAlignment;
  int deviceOverlap;
  int multiProcessorCount;
  int kernelExecTimeoutEnabled;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int maxTexture1D;
  int maxTexture1DMipmap;
  int maxTexture1DLinear;
  int maxTexture2D[2];
  int maxTexture2DMipmap[2];
  int maxTexture2DLinear[3];
  int maxTexture2DGather[2];
  int maxTexture3D[3];
  int maxTexture3DAlt[3];
  int maxTextureCubemap;
  int maxTexture1DLayered[2];
  int maxTexture2DLayered[3];
  int maxTextureCubemapLayered[2];
  int maxSurface1D;
  int maxSurface2D[2];
  int maxSurface3D[3];
  int maxSurface1DLayered[2];
  int maxSurface2DLayered[3];
  int maxSurfaceCubemap;
  int maxSurfaceCubemapLayered[2];
  size_t surfaceAlignment;
  int concurrentKernels;
  int ECCEnabled;
  int pciBusID;
  int pciDeviceID;
  int pciDomainID;
  int tccDriver;
  int asyncEngineCount;
  int unifiedAddressing;
  int memoryClockRate;
  int memoryBusWidth;
  int l2CacheSize;
  int persistingL2CacheMaxSize;
  int maxThreadsPerMultiProcessor;
  int streamPrioritiesSupported;
  int globalL1CacheSupported;
  int localL1CacheSupported;
  size_t sharedMemPerMultiprocessor;
  int regsPerMultiprocessor;
  int managedMemory;
  int isMultiGpuBoard;
  int multiGpuBoardGroupID;
  int hostNativeAtomicSupported;
  int singleToDoublePrecisionPerfRatio;
  int pageableMemoryAccess;
  int concurrentManagedAccess;
  int computePreemptionSupported;
  int canUseHostPointerForRegisteredMem;
  int cooperativeLaunch;
  int cooperativeMultiDeviceLaunch;
  size_t sharedMemPerBlockOptin;
  int pageableMemoryAccessUsesHostPageTables;
  int directManagedMemAccessFromHost;
  int maxBlocksPerMultiProcessor;
  int accessPolicyMaxWindowSize;
  size_t reservedSharedMemPerBlock;
  int hostRegisterSupported;
  int memoryPoolsSupported;
  int clusterLaunch;
  int unifiedFunctionPointers;
  char luid[8];
  unsigned int luidDeviceNodeMask;
  int ipcEventSupported;
  int timelineSemaphoreInteropSupported;
  int gpuDirectRDMASupported;
  unsigned int gpuDirectRDMAFlushWritesOptions;
  int gpuDirectRDMAWritesOrdering;
  unsigned int memoryPoolSupportedHandleTypes;
  int deferredMappingCudaArraySupported;
  int sparseCudaArraySupported;
  int maxBlocksPerMultiprocessor;
};

enum cudaDeviceP2PAttr {
  cudaDevP2PAttrPerformanceRank = 1,
  cudaDevP2PAttrAccessSupported = 2,
  cudaDevP2PAttrNativeAtomicSupported = 3,
  cudaDevP2PAttrCudaArrayAccessSupported = 4
};

enum cudaAccessProperty {
  cudaAccessPropertyNormal = 0,
  cudaAccessPropertyStreaming = 1,
  cudaAccessPropertyPersisting = 2
};
struct cudaAccessPolicyWindow {
  void* base_ptr;
  size_t num_bytes;
  float hitRatio;
  enum cudaAccessProperty hitProp;
  enum cudaAccessProperty missProp;
};
enum cudaLaunchAttributeID {
  cudaLaunchAttributeIgnore = 0,
  cudaLaunchAttributeAccessPolicyWindow = 1,
  cudaLaunchAttributeCooperative = 2,
  cudaLaunchAttributeSynchronizationPolicy = 3,
  cudaLaunchAttributePriority = 8
};
typedef enum cudaLaunchAttributeID cudaStreamAttrID;
typedef enum cudaLaunchAttributeID cudaKernelNodeAttrID;
#define cudaStreamAttributeAccessPolicyWindow cudaLaunchAttributeAccessPolicyWindow
#define cudaStreamAttributeSynchronizationPolicy cudaLaunchAttributeSynchronizationPolicy
#define cudaStreamAttributePriority cudaLaunchAttributePriority
#define cudaKernelNodeAttributeAccessPolicyWindow cudaLaunchAttributeAccessPolicyWindow
union cudaLaunchAttributeValue {
  char pad[64];
  struct cudaAccessPolicyWindow accessPolicyWindow;
  int cooperative;
  int priority;
};
typedef union cudaLaunchAttributeValue cudaStreamAttrValue;
typedef union cudaLaunchAttributeValue cudaKernelNodeAttrValue;

struct cudaFuncAttributes {
  size_t sharedSizeBytes;
  size_t constSizeBytes;
  size_t localSizeBytes;
  int maxThreadsPerBlock;
  int numRegs;
  int ptxVersion;
  int binaryVersion;
  int cacheModeCA;
  int maxDynamicSharedSizeBytes;
  int preferredShmemCarveout;
};

// Opaque handles
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
typedef void (*cudaHostFn_t)(void* userData);
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void* userData);

#define cudaStreamLegacy ((cudaStream_t)0x1)
#define cudaStreamPerThread ((cudaStream_t)0x2)

// Flags
#define cudaStreamDefault 0x00
#define cudaStreamNonBlocking 0x01
#define cudaEventDefault 0x00
#define cudaEventBlockingSync 0x01
#define cudaEventDisableTiming 0x02
#define cudaEventInterprocess 0x04
#define cudaHostAllocDefault 0x00
#define cudaHostAllocPortable 0x01
#define cudaHostAllocMapped 0x02
#define cudaHostAllocWriteCombined 0x04
#define cudaHostRegisterDefault 0x00
#define cudaHostRegisterPortable 0x01
#define cudaHostRegisterMapped 0x02
#define cudaHostRegisterIoMemory 0x04
#define cudaHostRegisterReadOnly 0x08
#define cudaMemAttachGlobal 0x01
#define cudaMemAttachHost 0x02
#define cudaMemAttachSingle 0x04
#define cudaDeviceScheduleAuto 0x00
#define cudaDeviceScheduleSpin 0x01
#define cudaDeviceScheduleYield 0x02
#define cudaDeviceScheduleBlockingSync 0x04
#define cudaDeviceBlockingSync 0x04
#define cudaDeviceMapHost 0x08
#define cudaDeviceLmemResizeToMax 0x10
#define cudaCpuDeviceId ((int)-1)
#define cudaInvalidDeviceId ((int)-2)

#endif  // POLYKERNEL_PK_TYPES_H
