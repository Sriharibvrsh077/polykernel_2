# Changelog

## 0.3.0

* **PTX back end for NVIDIA GPUs.** `pkcc` now has its own compiler from CUDA C++ kernels to PTX,
  NVIDIA's documented GPU assembly language (`tools/pk_cparse.py`, `tools/pk_ptx.py`,
  `tools/pk_ptx_builtins.py`, math library in `tools/ptx_prelude.cu`). The PTX is embedded in the
  program and loaded through the NVIDIA driver API at run time (`runtime/pk_ptx.cpp`), so NVIDIA
  GPUs run the kernels with only the GPU driver installed. `PK_BACKEND=auto` tries it first;
  `PK_BACKEND=ptx` forces it. `-arch=sm_XX` sets the PTX target (default sm_60). The generated PTX
  is checked with `ptxas` at build time when one is installed.
* Covered by the PTX compiler: shared/constant/device memory, all atomics and scopes, barriers and
  fences, warp shuffles/votes/match/activemask, cooperative groups (block, tiles, coalesced groups,
  labeled/binary partitions, reduce/scan, grid queries), printf, malloc/free, `__ldg`, integer,
  conversion and rounded intrinsics, fast-math and full math functions. Template kernels, device
  lambdas and classes with methods still use the CPU/OpenCL back ends.
* **ptxsim** (`tools/ptxsim/ptxsim.cpp`): a stand-in for the NVIDIA driver that interprets PTX on
  the CPU, used to test the PTX back end without a GPU (`PK_CUDA_DRIVER_LIBRARY`). Warps run in
  lock step; misaligned accesses, deadlocks and endless loops are reported; registers and local
  memory start with garbage.
* `run_tests.sh` adds a PTX column when an NVIDIA driver (or ptxsim) is available. New test
  `t13_cg_partitions` (coalesced shfl_up/down and ballot, binary/labeled partitions, match).
* Programs without kernels (such as `deviceQuery`) now use the GPU back end when one is available.
* The device math library gained `frexp`, `modf` and `nextafter` (float and double).
* The Kaggle notebook runs every step on the PTX back end too (tests, bitlearn, SGEMM, cuda-samples).

## 0.2.0

* OpenCL C 1.2 devices (NVIDIA's OpenCL driver, many older and mobile GPUs). The translator
  now labels every pointer with its address space (global, shared or private) and emits one copy
  of a device function per address space of its pointer arguments, so the output no longer needs
  the generic address space of OpenCL C 2.0. `pkcc -v` reports whether each file also builds as
  OpenCL C 1.2. `PK_OPENCL_STD=CL1.2` forces the dialect for testing.
* `__device__`/`__constant__` variables are ordinary device buffers passed as hidden kernel (and
  device-function) arguments instead of program-scope variables. They work on OpenCL C 1.2,
  `cudaGetSymbolAddress` works on the OpenCL back end, and copies to symbols are plain buffer
  copies. Array sizes may be expressions.
* `cudaGetSymbolSize` is computed at the call site, so it works in CPU-only builds too.
* Atomics on private-memory pointers (needed by the private copy of device functions).
* The OpenCL runtime can use a vendor driver directly through the `cl_khr_icd` dispatch table
  when `libOpenCL` is not installed (`PK_OPENCL_ICD=1` forces it, `PK_OPENCL_VENDOR_LIBRARY`
  names the driver).
* `PK_CLANG=none` skips build-time validation of the OpenCL output.
* Verbose output names the OpenCL C dialect in use.
* New test `t12_device_vars`; `run_tests.sh` also runs every OpenCL test in forced OpenCL C 1.2
  mode and no longer needs `bc`.
* New `examples/bench_sgemm.cu` (naive, shared-memory tiled and register-tiled SGEMM; plain
  CUDA, builds with nvcc and pkcc).
* New `tools/make_kaggle_notebook.py`: a self-contained notebook that tests PolyKernel on a
  Kaggle GPU against nvcc.
* Verified on an NVIDIA Tesla T4: 26/26 tests, bitlearn bit-exact, 17/32 cuda-samples on the GPU,
  SGEMM at 95–100% of native CUDA.
* Fixes from the T4 run: an oversized block now returns `cudaErrorInvalidValue` as CUDA 12 does;
  no shift-count warning in `thread_block_tile<32>::mask()`; the benchmark warms the GPU up and
  reports the best of 3 rounds.

## 0.1.0

First prototype: `pkcc` driver, CPU back end (any ISA, Linux/Windows), OpenCL 2.0/3.0 back end,
cooperative groups and warp emulation, bitlearn port.
