# Roadmap: from prototype to "CUDA everywhere"

PolyKernel today has three back ends: CPU (any ISA), PTX (NVIDIA GPUs through the driver API,
with PolyKernel's own CUDA-to-PTX compiler) and OpenCL (any OpenCL 1.2, 2.0 or 3.0 device; the
translated code avoids the generic address space so NVIDIA's OpenCL C 1.2 accepts it).
This roadmap lists, in rough order of value, what is needed to cover more programs and more
hardware. Each item notes where it plugs in.

## 0. PTX compiler coverage (next)

`tools/pk_ptx.py` compiles C-style kernels and device functions, cooperative groups and the
intrinsics. What still sends a kernel to another back end, in order of how often it appears:
* **Templates.** Device function templates can be instantiated at each call site (argument
  types are known there: deduce the parameters, re-parse the template's tokens with them bound,
  cache the instance). Kernel templates also need the host pass to pass the instantiation to
  the launch (`k<float, 256><<<...>>>` gives it explicitly; deduced arguments need host types,
  i.e. the Clang front end of section 1).
* **Device lambdas** (`auto f = [&](int i) {...}; f(3);`): inline the body at the call with the
  defining scope visible (by-reference captures) or snapshotted (by-value captures).
* **Classes with methods / operator overloading** (`Vector<T>::push`, `float2 operator+`), and
  libcu++ objects such as `cuda::atomic_ref` (map the methods to atomics).
* **Inline PTX** (`asm volatile(...)`): PTX in, PTX out, so it can be passed through with the
  operands substituted.
* Grid-wide sync: `cuLaunchCooperativeKernel` exists in the driver; `grid.sync()` becomes a
  barrier on a global counter, as in CUDA's own implementation.
* Textures: `cuTexObjectCreate` and the `tex.2d` PTX instruction, both documented.

Performance work after the first GPU measurements: vector loads/stores (`ld.global.v4`) for
`float4`/`int4`, keeping more aggregates in registers, and `.nc` loads for `const __restrict__`
pointers. (The driver already caches its JIT output per GPU, so start-up costs are paid once.)



## 1. Robustness of the translator (highest value)

The host and device passes rewrite tokens of preprocessed C++. That is fast and needs no
compiler internals, but it is heuristic. The long-term fix is a **Clang-based front end**.
Clang already parses CUDA (`-x cuda`). A small Clang tool would:
* replace `<<<...>>>` using the real AST (exact kernel types, overloads, templates);
* emit device code from the AST instead of from tokens, so templates can be instantiated
  for the OpenCL path (each instantiation becomes one OpenCL kernel).

**Constant memory.** `__constant__` variables are currently `__global` buffers. Marking the
read-only ones `__constant` in OpenCL would use the GPU's constant cache (broadcast reads, as in
CUDA); the address-space inference already tracks where each pointer comes from, so it needs a
third label and one more device-function copy.

## 2. Warp intrinsics on GPUs

Shuffles, votes and tiled groups currently force the CPU back end. Options:
* `cl_khr_subgroups` + `cl_khr_subgroup_shuffle` with a required sub-group size of 32
  (`cl_intel_required_subgroup_size`: Intel GPUs, PoCL);
* emulation in local memory for convergent code. Most reductions are convergent, but that
  has to be proven per call site.
* on AMD hardware with 64-wide wavefronts, warp size must be reported as 64. CUDA code that
  hard-codes 32 then needs the local-memory emulation.

## 3. More GPU back ends

| back end | hardware | notes |
|---|---|---|
| SPIR-V for Vulkan compute | almost every GPU incl. mobile, Apple via MoltenVK | biggest reach. Lower the OpenCL C (or Clang's AST) to SPIR-V with `clspv`; the runtime mirrors pk_opencl.cpp on Vulkan buffers |
| Metal | Apple GPUs (OpenCL is deprecated and 1.2-only on macOS) | translate to Metal Shading Language: close to the current OpenCL C output; address spaces map 1:1 |
| HIP | AMD GPUs (ROCm) | HIP *is* CUDA with renamed functions; emit HIP source and use the HIP runtime directly |
| SYCL / Level Zero | Intel GPUs | through SPIR-V, or reuse the OpenCL path (Intel's OpenCL driver is first class) |
| WebGPU (WGSL) | browsers | same structure as Metal |

The runtime already separates the back ends (`pk_cpu.cpp`, `pk_ptx.cpp`, `pk_opencl.cpp`). A new
back end implements malloc/free/memcpy/memset/launch/symbols and plugs into `decide_backend()`.
The PTX compiler's front end (`pk_cparse.py`: a typed syntax tree of the device code) is also the
natural starting point for the Metal, WGSL and SPIR-V back ends above, instead of the
token-level OpenCL translation.

## 4. Missing API areas

* **Textures/surfaces:** CPU: a sampler implementation (address modes, filtering,
  normalized coordinates) over linear memory. OpenCL: `image2d_t` + `sampler_t`.
* **`cuda_fp16.h`:** `__half`/`__half2` as a 16-bit storage type with float arithmetic on
  CPUs (native `_Float16` where available); OpenCL `half` with `cl_khr_fp16`.
* **Cooperative launch:** grid-wide sync on the CPU by running all blocks concurrently
  (as many fibers as needed). On OpenCL it needs persistent-thread scheduling.
* **CUB / Thrust:** both are open source (NVIDIA CCCL) and Thrust has CPU back ends
  (CPP/OpenMP/TBB). Building them against PolyKernel's headers is mostly configuration.
  CUB's warp/block primitives map onto the existing warp emulation.
* **cuBLAS/cuFFT/cuRAND:** wrappers over portable libraries (OpenBLAS/BLIS, FFTW/pocketfft,
  CLBlast/VkFFT on GPUs). That is the approach SCALE and HIP take.
* Device-side `assert` returning `cudaErrorAssert`; CUDA graphs (record and replay);
  memory pools; multi-device support for several OpenCL devices.

## 5. Running *binaries* (the ZLUDA approach) - deliberately out of scope

Running already-compiled CUDA programs means implementing the driver API, and translating
the PTX (or SASS) embedded in the binary. Two cautions:
* **Legal:** NVIDIA's EULA forbids reverse engineering toolkit output in order to target
  non-NVIDIA platforms. PTX is a documented ISA, but extracting and translating the embedded
  code is exactly the activity the clause targets. Get legal advice before going there.
* **Technical:** it is a much larger project (ZLUDA has needed years). Source compatibility
  covers every program whose source you have.

## 6. Semantics the CPU back end does not reproduce yet

* **Implicit warp lockstep.** Lanes of a warp run one after another between barriers, so
  warp-synchronous code without `__syncwarp()` (e.g. NVIDIA's `dxtc` sample:
  `s[i] += s[i ^ 8];`) computes different results. Running the 32 lanes of a warp as SIMD
  lanes (ISPC-style lowering, see Performance) would reproduce lockstep behaviour and also
  make the CPU back end much faster.
* **Dynamic parallelism.** A kernel launching kernels currently gets `cudaErrorNotSupported`.
  Child grids could be queued and run when the parent block reaches
  `cudaDeviceSynchronize()` or finishes; that needs the parent's fiber to suspend.
* **Device-side `assert`.** It aborts the process; it could instead stop the grid and make
  the next synchronizing call return `cudaErrorAssert`.

## 7. Performance

* CPU: vectorize loop-mode kernels across threads (a kernel with no barriers is a
  data-parallel loop: compile it with `#pragma omp simd`, or generate SIMD with ISPC-style
  lowering). Use cheaper hand-written context switches (asm) for fiber mode. Pin workers.
* OpenCL: asynchronous copies with events, multiple queues for streams, sub-buffers instead
  of offset parameters, real event timing via `CL_QUEUE_PROFILING_ENABLE`.
