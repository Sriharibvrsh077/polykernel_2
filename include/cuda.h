// PolyKernel: only the runtime API is implemented; this header exists so that
// programs which include <cuda.h> for CUDA_VERSION still compile.
#ifndef POLYKERNEL_CUDA_H
#define POLYKERNEL_CUDA_H
#include "cuda_runtime.h"
typedef int CUdevice;
typedef enum cudaError_enum { CUDA_SUCCESS = 0, CUDA_ERROR_INVALID_VALUE = 1, CUDA_ERROR_NOT_SUPPORTED = 801 } CUresult;
#endif
