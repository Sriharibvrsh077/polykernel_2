# PolyKernel

**Compile unmodified CUDA programs for any CPU architecture, run their kernels on NVIDIA GPUs
through PolyKernel's own PTX compiler, and on other GPUs through OpenCL.**

PolyKernel is a clean-room, source-compatible implementation of the CUDA runtime API and
the CUDA C++ kernel language. `pkcc` is a drop-in replacement for `nvcc`: it compiles your
`.cu` files with any C++17 compiler (GCC or Clang). It also compiles your kernels into
**PTX**, NVIDIA's documented GPU assembly language, which the program hands to the NVIDIA
driver at run time, so NVIDIA GPUs run them without the CUDA toolkit. An OpenCL C translation
is embedded as well, for AMD, Intel, ARM Mali, Qualcomm Adreno and other OpenCL GPUs. If no GPU
is present, the kernels run on the CPU's cores.

```
$ bin/pkcc -o hello examples/hello.cu && ./hello
ran on: PolyKernel CPU (x86_64, 2 threads)

$ bin/pkcc --target=aarch64-linux-gnu -o hello-arm64 examples/hello.cu    # cross-compile
$ bin/pkcc --target=x86_64-w64-mingw32 -o hello.exe examples/hello.cu     # Windows
$ PK_BACKEND=ptx ./hello                                                  # NVIDIA GPU (driver only)
$ PK_BACKEND=opencl ./hello                                               # OpenCL device
ran on: <your GPU> (OpenCL via PolyKernel)
```

## About "reverse engineering"

PolyKernel does **not** disassemble, decompile or patch NVIDIA's compiler, driver, libraries
or compiled CUDA binaries. NVIDIA's CUDA EULA forbids reverse engineering CUDA toolkit output
in order to run it on non-NVIDIA hardware (which is why binary translators such as ZLUDA are
legally contested). PolyKernel avoids that entirely:

* it works from **source code**, the same way AMD HIP, Spectral SCALE and chipStar do;
* every header and runtime file was written from the **public CUDA programming guide and
  runtime API reference** (function names, types, error codes and documented behaviour);
* no NVIDIA header, library or binary is needed, read or shipped;
* the PTX back end generates PTX from NVIDIA's public **PTX ISA** document and loads it through
  the public **driver API** of the installed GPU driver, the same documented route that
  compilers such as Clang, Triton and Numba use. If NVIDIA's `ptxas` assembler happens to be
  installed, `pkcc` runs it on the generated PTX as an extra check; it is never required.

The API surface is what makes this work. CUDA programs talk to `cudaMalloc`, `<<<...>>>`,
`threadIdx`, `__syncthreads()` and so on. Reimplementing that interface is enough to run them
elsewhere. (This is a description of the project's design, not legal advice.)

## How it works

```
                 ┌─────────────── pkcc ────────────────┐
  app.cu ──► host pass:   c++ -E  ─► rewrite  ─► host compiler ─► app.o ─┐
         │   (<<<>>> → pk::launch_k, __shared__ → per-block storage)     │
         └─► device pass: c++ -E (system headers stubbed) ─► CUDA→OpenCL C│
             (validated with clang, embedded into app.o as a string)      │
                                                                          ▼
                                           app = app.o + libpolykernel.a (built per target)

         └─► PTX pass:    c++ -E (__CUDA_ARCH__ = target) ─► CUDA→PTX (own compiler)
             (checked with ptxas if installed, embedded into app.o as a string)

  at run time:  NVIDIA driver found and every launched kernel compiled to PTX?
                   yes → PTX back end      (libcuda is loaded dynamically, the driver JIT-compiles the PTX)
                   no  → OpenCL GPU found and every launched kernel translated?
                           yes → OpenCL back end   (libOpenCL is loaded dynamically)
                           no  → CPU back end      (worker threads + fibers)
```

**CPU back end** (any architecture with a C++17 compiler):
* The blocks of a launch are spread over one worker thread per core.
* Within a block, CUDA threads run as a simple loop when the kernel has no barriers. When it
  uses `__syncthreads()` or warp intrinsics, each thread becomes a lightweight fiber with its
  own stack, and every barrier switches to the next fiber. That gives exact CUDA barrier
  semantics, including threads that exit early.
* `__shared__` variables become worker-thread-local storage. Since a block never migrates
  between workers, that is exactly one copy per block.
* Warp shuffles, votes, `__syncwarp`, match/reduce intrinsics and cooperative groups (block,
  tiles ≤ 32, coalesced and labeled groups, `cg::reduce`/scan) are emulated with 32-lane
  warps, as on NVIDIA. `__activemask()`/`coalesced_threads()` return the lanes that actually
  reach that call site together, so warp-aggregated atomics in divergent code work.
* Kernels are compiled with `__CUDA_ARCH__` defined (800 by default, or the value implied by
  `-arch=sm_XX`), as nvcc does for device code, so `#if __CUDA_ARCH__ >= 700` paths are kept.
* A block whose threads wait at barriers that can never complete stops with a "deadlock"
  message instead of hanging.
* Atomics are real hardware atomics, because blocks run in parallel.

**PTX back end** (NVIDIA GPUs, any driver from the last several years; no CUDA toolkit):
* `pkcc` contains a compiler from CUDA C++ kernels to PTX (`tools/pk_cparse.py` parses the
  preprocessed device code into a typed syntax tree, `tools/pk_ptx.py` generates PTX). Device
  functions are inlined; scalars live in virtual registers that the driver's register allocator
  assigns; small arrays and structs indexed with constants are split into registers, everything
  else goes to local memory. Pointers carry a provenance analysis (global, shared, local,
  constant), so most loads and stores use the specific state space instead of generic addressing.
  `#pragma unroll` with constant trip counts is honoured.
* Covered: the built-in variables, shared memory (static and dynamic), `__constant__` and
  `__device__` variables, all atomics (also `_block`/`_system` scopes), `__syncthreads*`, fences,
  warp shuffles/votes/`__activemask`/`__syncwarp`/match, `__ldg` and cache hints, integer and
  bit intrinsics, the rounded `__fadd_rn`-style intrinsics and conversions, fast-math intrinsics,
  `printf` (through the driver's `vprintf`), `malloc`/`free`, and cooperative groups (block, tiles,
  coalesced groups, `labeled_partition`/`binary_partition`, `cg::reduce` and scans, grid queries).
  Math functions without a PTX instruction (`expf`, `sin`, `pow`, `erf`, `lgamma`, `frexp`, ...)
  come from `tools/ptx_prelude.cu`, CUDA C compiled by the same compiler.
* Floating-point results follow CUDA's rules: `+`/`*` may be contracted into FMA as nvcc does by
  default, division and `sqrt` are IEEE-rounded, and `-arch=sm_XX` picks the PTX target
  (default `sm_60`, which every driver can JIT for Pascal and newer GPUs).
* At run time the runtime loads `libcuda` (`nvcuda.dll` on Windows), JIT-loads one module per
  `.cu` file, copies the initial values of `__device__`/`__constant__` variables into the
  module's globals, and launches kernels with `cuLaunchKernel`. Device pointers are real GPU
  addresses, so pinned/mapped/managed memory and pointer arithmetic work as with CUDA.
* A kernel the compiler cannot handle yet (template kernels, device lambdas, classes with
  methods) stays CPU/OpenCL-only; `pkcc -v` says why. If any launched kernel lacks PTX, the
  program uses the next back end, as with OpenCL.
* `tools/ptxsim/` is a small stand-in for the NVIDIA driver that interprets PTX on the CPU
  (threads, CTA barriers, lock-step warps with shuffles/votes/match, atomics, alignment checks,
  garbage-initialised registers). The whole test suite runs on it, which is how the PTX back end
  is tested on machines without an NVIDIA GPU:
  ```
  c++ -std=c++17 -O2 -shared -fPIC -frounding-math -ffp-contract=off -pthread \
      -o libptxsim.so tools/ptxsim/ptxsim.cpp
  PK_CUDA_DRIVER_LIBRARY=$PWD/libptxsim.so tests/run_tests.sh
  ```

**OpenCL back end** (GPUs and accelerators of any vendor):
* Kernels are translated to OpenCL C that builds as 1.2, 2.0 or 3.0, whichever the driver
  offers. OpenCL C 1.2 (what NVIDIA's driver accepts, and many older or mobile GPUs) has no
  generic address space, so the translator works out whether each pointer refers to global,
  shared or private memory, labels it, and emits one copy of a device function per memory space
  of its pointer arguments. Unsupported constructs make that kernel CPU-only; they never break
  the build. `pkcc -v` lists what was translated, why anything was not, and whether the result
  also builds as OpenCL C 1.2.
* `cudaMalloc` returns addresses from a reserved, unmapped range of the host address space.
  The runtime maps those addresses, including pointer arithmetic on them, to (buffer, offset)
  pairs, so host code using device pointers works unchanged.
* `__constant__`/`__device__` variables live in ordinary device buffers that are passed to the
  kernels (and device functions) using them as hidden arguments, so they work on OpenCL C 1.2
  too. `cudaMemcpyToSymbol`/`FromSymbol` and `cudaGetSymbolAddress` work on them. Dynamic shared
  memory, managed memory and function-pointer launches are supported.
* `libOpenCL` (the standard ICD loader) is loaded at run time if present. Without it the runtime
  talks to a vendor driver such as `libnvidia-opencl.so.1` directly through the standard
  `cl_khr_icd` dispatch table, so containers that ship only the GPU driver work too.
* The back end is chosen once per process. If any kernel the program can launch lacks a
  translation, the whole program uses the CPU, so device memory never has to move between
  back ends.

## Requirements

* Python 3.8+ for the `pkcc` driver and translator. `pkcc` uses `python3.14` when it is
  installed (tested with 3.14.6, also 3.12 and 3.13); `PK_PYTHON=/path/to/python` overrides.
* A C++17 compiler for each target: `g++`/`clang++`, or `<triple>-g++` for cross builds
* Optional: `clang` with OpenCL support at build time (validates the generated OpenCL C, so a
  kernel the driver would reject is left CPU-only instead of sending the program to the CPU;
  tested with clang 14, 16 and 18; `PK_CLANG=none` skips it)
* Optional: an OpenCL 1.2 or newer driver at run time
* Optional: an NVIDIA GPU driver at run time for the PTX back end (`libcuda.so.1`, which every
  NVIDIA driver installs; the CUDA toolkit is not needed). If `ptxas` is on the `PATH` (or in
  `/usr/local/cuda/bin`) at build time, `pkcc` uses it to check the generated PTX
  (`PK_PTXAS=/path/to/ptxas` picks one, `PK_PTXAS=none` skips the check).

The runtime library is built automatically, once per compiler/target, into `lib/<triple>/`.

## Usage

`pkcc --version` also shows which Python runs it. `pkcc` accepts the usual nvcc options (`-c`, `-o`, `-I`, `-D`, `-L`, `-l`, `-O*`, `-g`,
`-std=`, `-x cu`, `-Xcompiler`, `-Xlinker`, `-ccbin`, `-M*`). `-arch=sm_XX` sets the PTX target
and `__CUDA_ARCH__`; other GPU-specific options are ignored (`-gencode`, `-rdc`,
`--use_fast_math`, `-lineinfo`, ...). Additional options:

| option | meaning |
|---|---|
| `--target=<triple>` | cross-compile (uses `<triple>-g++`, or `clang++ --target`) |
| `--pk-backend=cpu` | skip OpenCL and PTX code generation |
| `--pk-cuda-arch=N\|none` | value of `__CUDA_ARCH__` while compiling kernels (default 800; `-arch=sm_XX` sets it too) |
| `-v` | show commands, translated kernels and the reasons kernels stay CPU-only |
| `--keep` | keep the translated C++ (`.pk.ii`), OpenCL C (`.cl`) and PTX (`.ptx`) next to the object |

Run-time environment variables:

| variable | meaning |
|---|---|
| `PK_BACKEND=auto\|cpu\|ptx\|opencl` | `auto` (default) tries the NVIDIA driver (PTX), then an OpenCL GPU/accelerator, then the CPU; `ptx` (or `cuda`) and `opencl` force one GPU path (`opencl` also accepts OpenCL CPU devices) |
| `PK_VERBOSE=1` | print the chosen back end and why |
| `PK_NUM_THREADS=N` | CPU worker threads (default: all cores) |
| `PK_OPENCL_DEVICE=i` | pick an OpenCL device by index |
| `PK_CUDA_DEVICE=i` | pick an NVIDIA GPU by index (PTX back end) |
| `PK_CUDA_DRIVER_LIBRARY=path` | driver library to load instead of `libcuda.so.1` (e.g. `ptxsim`) |
| `PK_NO_PTX=1` | skip the PTX back end in `auto` mode |
| `PK_STACK_SIZE=256K` | fiber stack size per CUDA thread |
| `PK_FORCE_FIBERS=1` | barrier support for every kernel (needed only when a barrier sits in a device function in another `.cu` file) |
| `PK_SYNC_LAUNCH=1` | wait for every GPU kernel and report its errors at the launch (debugging) |
| `PK_OPENCL_STD=CL1.2` | force an OpenCL C dialect (default: the newest the device supports) |
| `PK_OPENCL_ICD=1` | skip `libOpenCL` and use the vendor driver directly |
| `PK_OPENCL_VENDOR_LIBRARY=path` | vendor driver to use when there is no `libOpenCL` |
| `PK_OPENCL_BUILD_OPTIONS=...` | extra options for the OpenCL compiler |

## Test results

`tests/run_tests.sh <targets...>` builds and runs 14 self-checking programs (15 sources).
They cover vector ops, tiled matrix multiply, reductions, histograms with shared/global
atomics, constant memory, `__device__`/`__constant__` variables of every shape, structs, 3-D grids, the runtime API (streams, events, pinned and
managed memory, pitched copies, `cudaLaunchKernel`, error reporting), templates, device
lambdas, function pointers, warp intrinsics, cooperative groups (including coalesced,
labeled and binary partitions in divergent code), multi-file programs, the
C++ features the OpenCL translation supports, and a CUDA port of the bitlearn Tsetlin
Machine kernels that must reproduce bitlearn's own results bit for bit
([examples/bitlearn](examples/bitlearn/README.md)). `pkcc` ran under Python 3.14.6 with
development mode on and every warning turned into an error.

| target | how it ran | result |
|---|---|---|
| x86-64 Linux, CPU back end | native | 14/14 |
| x86-64 Linux, OpenCL back end | PoCL (software OpenCL driver), OpenCL C 3.0 | 14/14 (4 correctly fall back to the CPU: templates, warp intrinsics, coalesced groups) |
| x86-64 Linux, OpenCL back end | PoCL forced to OpenCL C 1.2 (`PK_OPENCL_STD=CL1.2`) | 14/14 (same 4 fall back) |
| x86-64 Linux, PTX back end | ptxsim in place of the NVIDIA driver; every module also accepted by NVIDIA's `ptxas` 12.9 | 14/14 (1 correctly falls back to the CPU: template kernels) |
| ARM64 (aarch64) Linux | cross-compiled, QEMU | 14/14 |
| ARMv7 (32-bit, hard-float) Linux | cross-compiled, QEMU | 14/14 |
| RISC-V 64 Linux | cross-compiled, QEMU | 14/14 |
| IBM Z s390x Linux (big-endian) | cross-compiled, QEMU | 14/14 |
| POWER ppc64le Linux | cross-compiled, QEMU | 14/14 |
| x86-64 Windows | MinGW cross-compiled, Wine | 14/14 |

The OpenCL runs also pass with validation turned off (`PK_CLANG=none`), with the driver used
directly instead of `libOpenCL` (`PK_OPENCL_ICD=1`), and with GCC 11 as the host compiler.
The test sources, the bitlearn port and the benchmark also compile unchanged as real CUDA for
`sm_60` (clang's CUDA mode with NVIDIA's headers and `ptxas`), except `t11` and `t13`, which use
`labeled_partition` and `__match_*_sync` (compute capability 7.0+ with nvcc; PolyKernel's PTX
compiler emulates them on older GPUs).

Unmodified NVIDIA [cuda-samples](https://github.com/NVIDIA/cuda-samples) (BSD-3), built
with `pkcc -O2 -I Common` and no edits: **32 of 130** build and pass on the CPU back end
(17 of them also run their kernels on OpenCL, in both the 3.0 and the forced 1.2 dialect;
on the PTX back end in ptxsim, 20 run their kernels as PTX and pass, `deviceQuery` reports the
GPU, 5 compile to PTX but are too heavy for the interpreter, and 6 still need templates,
device lambdas or classes in kernels), 2 more are correct but exceed the 60 s
limit on 2 cores, and 2 correctly waive themselves (they need two GPUs / a persistent L2
cache). Nearly all of the rest need things outside this project's scope so far: graphics
interop (25), NVIDIA libraries such as cuBLAS/CUB/Thrust (18), textures (13) and the driver
API (12). Details: [docs/SAMPLES.md](docs/SAMPLES.md).

**Verified on an NVIDIA Tesla T4** with PolyKernel 0.2.0's OpenCL back end (Kaggle, driver
580.178.04, CUDA 13.0, NVIDIA OpenCL 3.0 with OpenCL C 1.2): the test suite passes 26/26 (10 programs run their kernels on the T4 through OpenCL,
3 fall back to the CPU as expected), the bitlearn port matches bitlearn bit for bit on the GPU, and
17 of the 32 cuda-samples above run on the T4 unmodified (the other 15 pass on the CPU; 0 failures).
The same programs built with `nvcc` for `sm_75` pass 13/13 and 32/32 as the reference.

Speed on the T4, same source built with `nvcc` and with `pkcc` (GFLOP/s, single precision;
each kernel warmed up for 300 ms, best of 3 rounds):

| SGEMM kernel | N=1024 CUDA / PolyKernel | N=2048 | N=4096 |
|---|---|---|---|
| naive | 453 / 449 (99%) | 392 / 382 (97%) | 382 / 371 (97%) |
| shared-memory tiles | 736 / 701 (95%) | 601 / 582 (97%) | 581 / 552 (95%) |
| register tiles | 1708 / 1704 (100%) | 1685 / 1680 (100%) | 1555 / 1504 (97%) |

PolyKernel's event timing waits for the GPU and then reads the host clock, so its times
include a little host overhead. The bitlearn port runs equally fast either way (inference
0.09 ms with shared memory; training ≈40–50 ms per epoch for both, slower than the 3 ms on 4 CPU
cores because the training kernel uses one GPU thread per class, as bitlearn's own code does).
[tools/make_kaggle_notebook.py](tools/make_kaggle_notebook.py) builds the notebook that ran this.
The PTX back end (0.3.0) has not run on a real GPU yet: the same notebook now runs every step on it
as well, next to the OpenCL path and nvcc. AMD, Intel and ARM GPU drivers have not been tried yet.

## What is supported

* Runtime API: device management/properties/attributes, `cudaMalloc*`/`Free*`, all
  `cudaMemcpy*`/`cudaMemset*` variants including 2-D and async, pinned/mapped/managed memory,
  symbols, streams, events, callbacks/host functions, `cudaLaunchKernel`, occupancy helpers,
  function attributes, error API, pointer attributes.
* Kernel language: `__global__`/`__device__`/`__host__`/`__shared__` (static and `extern`
  dynamic)/`__constant__`/`__managed__`, `<<<grid, block, shmem, stream>>>`, the built-in
  variables, vector types, `__syncthreads*`, `__threadfence*`, all atomics, warp
  shuffle/vote/match/reduce intrinsics, CUDA math and integer intrinsics, `printf`,
  cooperative groups (block, tiles, coalesced, reduce/scan), libcu++ basics (`cuda::std::`,
  `cuda::atomic_ref`, `cuda::ceil_div`).
* Full C++17 inside kernels on the CPU back end: templates, lambdas, classes, functors.

**PTX compiler subset:** C-style kernels and `__device__` functions (inlined), structs/enums/
typedefs/namespaces, file-scope constants, `__constant__`/`__device__` variables, static and
dynamic shared memory, all atomics, warp intrinsics, cooperative groups (block, tiles, coalesced
groups, labeled/binary partitions, reduce/scan, grid queries), `printf`, `malloc`/`free`, the
CUDA math library, kernel parameters up to 32 KB (Volta and newer). CPU/OpenCL-only for now:
template kernels and device function templates, device lambdas, classes with methods and
operator overloading, `__managed__` variables, inline PTX, textures and libcu++ objects.

**OpenCL translation subset:** C-style kernels and `__device__` functions, structs/enums/
typedefs, file-scope constants, `__constant__`/`__device__` variables (scalars, arrays,
structs; not pointers or constructor-initialised objects), atomics, math, vector
types, `static_cast` and functional casts, default kernel arguments, reference parameters of
device functions, block-level cooperative groups, function-pointer launches.
CPU-only for now: templates, classes with methods, warp intrinsics, tiled/coalesced groups,
taking the address of a vector component, textures and libcu++.

## Not supported (yet)

Texture/surface objects, `cuda_fp16.h`, the CUDA driver API (`cu*`) in user code, inline PTX
assembly in kernels, NVRTC, grid-wide sync (cooperative launch), dynamic parallelism, IPC, graphs, memory pools,
multi-GPU, and the NVIDIA libraries (cuBLAS, cuFFT, cuDNN, CUB, Thrust). A device-side
`assert` aborts the process instead of returning `cudaErrorAssert`; a kernel launched from
device code returns `cudaErrorNotSupported`.

**Implicit warp lockstep.** Code that exchanges data between lanes of a warp through shared
memory *without* `__syncwarp()`, relying on all 32 lanes executing each instruction together
(for example `s[i] += s[i ^ 8];` with no barrier in between), is a data race according to
CUDA's own rules since compute capability 7.0, but often still works on NVIDIA hardware (and on
PolyKernel's PTX back end, which runs on that hardware). The CPU back end runs a warp's lanes
one after another between barriers, so such code computes different results there. Add `__syncwarp()` between the read and the write. See
[docs/ROADMAP.md](docs/ROADMAP.md) for how each of these can be added.

## Layout

```
bin/pkcc                 nvcc-compatible driver
tools/pk_host.py         host pass (C++ rewriting)
tools/pk_ocl.py          device pass (CUDA C++ -> OpenCL C), tools/ocl_prelude.cl
tools/pk_cparse.py       device code parser (typed syntax tree) used by the PTX compiler
tools/pk_ptx.py          CUDA -> PTX compiler; tools/pk_ptx_builtins.py (intrinsics), tools/ptx_prelude.cu (math)
tools/ptxsim/ptxsim.cpp  PTX interpreter that stands in for the NVIDIA driver in tests
include/                 cuda_runtime.h & friends, cooperative_groups.h, cuda/ (libcu++ subset)
runtime/                 libpolykernel: pk_runtime.cpp (API), pk_cpu.cpp, pk_ptx.cpp (NVIDIA driver API), pk_opencl.cpp
third_party/khronos/     OpenCL headers (Apache-2.0, Khronos Group)
tests/                   self-checking test programs and run_tests.sh
examples/hello.cu
examples/bench_sgemm.cu  SGEMM benchmark (naive / shared-memory tiles / register tiles), plain CUDA
examples/bitlearn/       bitlearn's Tsetlin Machine kernels in CUDA, verified against bitlearn
tools/make_kaggle_notebook.py   builds the GPU test notebook (embeds the source tarball)
```
