# NVIDIA cuda-samples, unmodified, built with pkcc

Source: [github.com/NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) (BSD-3-Clause), categories 0_Introduction, 1_Utilities, 2_Concepts_and_Techniques, 3_CUDA_Features, 5_Domain_Specific and 6_Performance. Each sample was built from its own sources with `pkcc -O2 -I Common` (no edits, no build files) and run in its directory with a 60 s limit, on a 2-core x86-64 VM. Passing samples were run again with `PK_BACKEND=opencl` (PoCL). `pkcc` ran on Python 3.14.6.

| result | samples |
|---|---|
| build and run correctly (CPU back end) | 32 of 130 |
| ... of those, kernels also ran on the OpenCL back end | 17 |
| build, but do not complete correctly | 9 |
| do not build | 89 |

## Passing samples

| sample | OpenCL back end |
|---|---|
| 0_Introduction/asyncAPI | yes, kernels ran on OpenCL |
| 0_Introduction/simpleAtomicIntrinsics | CPU fallback: simpleAtomicIntrinsics.cu: kernel 'cas_atomic_cuda' has no OpenCL translation |
| 0_Introduction/simpleCallback | yes, kernels ran on OpenCL |
| 0_Introduction/simpleCooperativeGroups | CPU fallback: simpleCooperativeGroups.cu: kernel 'cgkernel' has no OpenCL translation |
| 0_Introduction/simpleMultiCopy | yes, kernels ran on OpenCL |
| 0_Introduction/simplePrintf | yes, kernels ran on OpenCL |
| 0_Introduction/simpleStreams | yes, kernels ran on OpenCL |
| 0_Introduction/simpleVoteIntrinsics | CPU fallback: simpleVoteIntrinsics.cu: kernel 'VoteAllKernel2' has no OpenCL translation |
| 0_Introduction/simpleZeroCopy | yes, kernels ran on OpenCL |
| 0_Introduction/systemWideAtomics | CPU fallback: systemWideAtomics.cu: kernel 'atomicKernel' has no OpenCL translation |
| 0_Introduction/vectorAdd | yes, kernels ran on OpenCL |
| 1_Utilities/deviceQuery | CPU fallback: no kernels were compiled with pkcc in this program |
| 2_Concepts_and_Techniques/convolutionSeparable | yes, kernels ran on OpenCL |
| 2_Concepts_and_Techniques/eigenvalues | CPU fallback: bisect_small.cu: kernel 'bisectKernel' has no OpenCL translation |
| 2_Concepts_and_Techniques/histogram | yes, kernels ran on OpenCL |
| 2_Concepts_and_Techniques/scalarProd | yes, kernels ran on OpenCL |
| 2_Concepts_and_Techniques/shfl_scan | CPU fallback: shfl_scan.cu: kernel 'shfl_intimage_rows' has no OpenCL translation |
| 2_Concepts_and_Techniques/sortingNetworks | yes, kernels ran on OpenCL |
| 2_Concepts_and_Techniques/threadFenceReduction | CPU fallback: threadFenceReduction.cu: launches a template kernel or a kernel through a function pointer (CPU-only feature) |
| 3_CUDA_Features/StreamPriorities | yes, kernels ran on OpenCL |
| 3_CUDA_Features/binaryPartitionCG | CPU fallback: binaryPartitionCG.cu: kernel 'oddEvenCountAndSumCG' has no OpenCL translation |
| 3_CUDA_Features/newdelete | CPU fallback: newdelete.cu: kernel 'complexVector' has no OpenCL translation |
| 3_CUDA_Features/warpAggregatedAtomicsCG | CPU fallback: warpAggregatedAtomicsCG.cu: kernel 'calculateMaxInEachBuckets' has no OpenCL translation |
| 5_Domain_Specific/BlackScholes | yes, kernels ran on OpenCL |
| 5_Domain_Specific/SobolQRNG | CPU fallback: sobol_gpu.cu: kernel 'sobolGPU_kernel' has no OpenCL translation |
| 5_Domain_Specific/binomialOptions | yes, kernels ran on OpenCL |
| 5_Domain_Specific/dwtHaar1D | yes, kernels ran on OpenCL |
| 5_Domain_Specific/fastWalshTransform | yes, kernels ran on OpenCL |
| 5_Domain_Specific/quasirandomGenerator | CPU fallback: quasirandomGenerator_kernel.cu: kernel 'inverseCNDKernel' has no OpenCL translation |
| 6_Performance/LargeKernelParameter | CPU fallback: LargeKernelParameter.cu: kernel 'kernelDefault' has no OpenCL translation |
| 6_Performance/alignedTypes | CPU fallback: alignedTypes.cu: launches a template kernel or a kernel through a function pointer (CPU-only feature) |
| 6_Performance/transpose | yes, kernels ran on OpenCL |

## Samples that build but do not complete

| sample | reason |
|---|---|
| 0_Introduction/matrixMul | correct (Result = PASS) but takes 103 s on 2 cores (exceeds the 60 s limit) |
| 0_Introduction/simpleAssert | device-side assert aborts the process (expected; documented limitation) |
| 0_Introduction/simpleAttributes | waived by the sample itself (needs persisting L2 cache) |
| 0_Introduction/simpleP2P | waived by the sample itself (needs two GPUs) |
| 2_Concepts_and_Techniques/scan | correct but slow: 100 iterations over 6.8M elements per configuration (exceeds the 60 s limit on 2 cores) |
| 3_CUDA_Features/cdpBezierTessellation | exits 0, but its device-side launches are refused (dynamic parallelism unsupported) |
| 3_CUDA_Features/cdpSimplePrint | dynamic parallelism (kernel launching kernels) is not supported |
| 3_CUDA_Features/cdpSimpleQuicksort | dynamic parallelism is not supported |
| 5_Domain_Specific/dxtc | relies on implicit warp lockstep (s[i] += s[i^8] without __syncwarp): documented limitation |

## Why the others do not build

| missing feature | samples |
|---|---|
| graphics/compute interop (OpenGL, EGL, Direct3D, Vulkan) | 25 |
| needs an NVIDIA library (cuBLAS/cuFFT/CUB/Thrust/NVRTC/...) or libcu++ feature not in the subset | 18 |
| texture / surface objects | 13 |
| CUDA driver API (cu*) | 12 |
| CUDA graphs | 4 |
| build setup (OpenMP flags / extra libraries) | 3 |
| memory pools / location API | 3 |
| IPC / MPI | 2 |
| newer device-attribute enum values | 2 |
| tensor-core matrix ops (mma.h / WMMA) | 2 |
| half / bfloat16 types (cuda_fp16.h) | 1 |
| inline PTX assembly | 1 |
| directed-rounding intrinsics | 1 |
| large cooperative-group tiles | 1 |
| cooperative launch / grid sync / clusters | 1 |

Most of these need features listed in [ROADMAP.md](ROADMAP.md) (NVIDIA libraries, textures, driver API, graphics interop, half precision).
