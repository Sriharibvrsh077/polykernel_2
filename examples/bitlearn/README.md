# bitlearn's Tsetlin Machine in CUDA, run everywhere with PolyKernel

`bitlearn_tm.cu` is a CUDA version of the two hot kernels of bitlearn 0.2.0, a
Tsetlin Machine library that learns with AND/XOR/NOT on 64-bit words (`bitlearn/_core.py`;
bitlearn itself is not part of this repository):

* **inference** (`class_sums`): vote total of every class for every sample: the
  non-empty clauses that fire (all included literals are 1), with signed weights;
  two variants, plain and with the include masks cached in dynamic `__shared__` memory;
* **training** (`fit_epoch`): one pass of Type I / Type II feedback over bit-sliced
  saturating counters, one CUDA thread per class, with bitlearn's xorshift64* streams.

It is checked **bit for bit** against bitlearn itself: `export_case.py` trains a bitlearn
model, then saves the model, the exact inputs of one more epoch (sample order, random
bit pool, per-class seeds) and bitlearn's outputs. `bitlearn_tm` recomputes everything
and compares.

```
# needs bitlearn 0.2.0 (+ numba, scikit-learn); tested with Python 3.14.6, numba 0.68.1
python export_case.py tm_case          # tm_case/ is already included here

../../bin/pkcc -O2 -o bitlearn_tm bitlearn_tm.cu
./bitlearn_tm tm_case
```

## Result (8x8 digits, 64 binary features, 10 classes x 200 clauses, 8-bit automata)

bitlearn test accuracy after 3 epochs: 92.70%. One further epoch changes about 28,000
of the 40,000 state words; the CUDA port must reproduce every one of them.

| target | back end | inference (3,970 vote totals) | training epoch, boosted | training epoch, weighted |
|---|---|---|---|---|
| x86-64 Linux | CPU | identical | identical | identical |
| x86-64 Linux | OpenCL (PoCL) | identical | identical | identical |
| ARM64 Linux | CPU (QEMU) | identical | identical | identical |
| ARMv7 Linux | CPU (QEMU) | identical | identical | identical |
| RISC-V 64 Linux | CPU (QEMU) | identical | identical | identical |
| s390x Linux (big-endian) | CPU (QEMU) | identical | identical | identical |
| ppc64le Linux | CPU (QEMU) | identical | identical | identical |
| x86-64 Windows | CPU (Wine) | identical | identical | identical |

All three kernels were translated to OpenCL C automatically. The training kernel uses
`double` for the feedback probabilities (as bitlearn does), so on OpenCL it needs a
device with `cl_khr_fp64`.

Timings in this setting are not a speed comparison with bitlearn's Numba code: the
training kernel deliberately uses one thread per class to reproduce bitlearn's random
streams exactly, which is right for checking correctness but leaves a GPU almost idle.
A GPU-oriented version would parallelise over clauses (bitlearn roadmap item 1).
