#!/usr/bin/env bash
# PolyKernel test matrix.
#   tests/run_tests.sh                      # native CPU (+ OpenCL if a driver is installed)
#   tests/run_tests.sh native aarch64-linux-gnu riscv64-linux-gnu ...
#   PK_PYTHON=python3.14 tests/run_tests.sh      # run pkcc with a specific Python interpreter
# Foreign targets are cross-compiled with <triple>-g++ and run under qemu-<arch>-static.
# The PTX back end is tested when an NVIDIA driver is installed, or with the simulator:
#   PK_CUDA_DRIVER_LIBRARY=/path/to/libptxsim.so tests/run_tests.sh
set -u
PK=$(cd "$(dirname "$0")/.." && pwd)
OUT=${PK_TEST_OUT:-/tmp/pk-tests}
mkdir -p "$OUT"
# shellcheck disable=SC2206
PKCC=(${PK_PYTHON:-} "$PK/bin/pkcc")
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(native)

single=(t01_vector_add t02_matmul_tiled t03_reduction t04_histogram t05_constant_struct t06_runtime_api
        t07_templates_lambdas t08_warp_coop t10_opencl_features t11_coalesced_groups t12_device_vars
        t13_cg_partitions)
# tests that rely on features the OpenCL path lacks (templates, warp intrinsics) and are expected to fall back
cpu_only=" t07_templates_lambdas t08_warp_coop t11_coalesced_groups t13_cg_partitions "
# extra tests expected to fall back to the CPU on OpenCL C 1.2 devices (e.g. NVIDIA drivers)
cpu_only_cl12=" "
# tests using features PolyKernel's PTX compiler does not handle yet (template kernels, device lambdas)
cpu_only_ptx=" t07_templates_lambdas "

qemu_for() {
  case "$1" in
    aarch64-*) echo qemu-aarch64-static ;;
    riscv64-*) echo qemu-riscv64-static ;;
    s390x-*) echo qemu-s390x-static ;;
    powerpc64le-*) echo qemu-ppc64le-static ;;
    powerpc64-*) echo qemu-ppc64-static ;;
    arm-*|armv7*) echo qemu-arm-static ;;
    i686-linux*) echo qemu-i386-static ;;
    *-mingw32|*-w64-*) for w in wine64 /usr/lib/wine/wine64 wine; do command -v "$w" >/dev/null 2>&1 && { echo "$w"; return; }; done; echo "" ;;
    *) echo "" ;;
  esac
}

have_opencl=0
# an ICD loader, or a driver PolyKernel can use directly
if [ -z "${PK_SKIP_OPENCL:-}" ] && { ldconfig -p 2>/dev/null | grep -qE 'libOpenCL\.so|libnvidia-opencl\.so|libpocl\.so|libamdocl'; }; then have_opencl=1; fi

# NVIDIA driver (or ptxsim, PolyKernel's PTX simulator, via PK_CUDA_DRIVER_LIBRARY) for the PTX back end
have_ptx=0
if [ -z "${PK_SKIP_PTX:-}" ]; then
  if [ -n "${PK_CUDA_DRIVER_LIBRARY:-}" ] && [ -f "${PK_CUDA_DRIVER_LIBRARY}" ]; then have_ptx=1
  elif ldconfig -p 2>/dev/null | grep -q 'libcuda\.so'; then have_ptx=1
  else
    ldp=${LD_LIBRARY_PATH:-}
    for d in ${ldp//:/ } /usr/local/nvidia/lib64 /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib/wsl/lib; do
      if ls "$d"/libcuda.so* >/dev/null 2>&1; then have_ptx=1; break; fi
    done
  fi
fi

pass=0; fail=0
results=()
for target in "${targets[@]}"; do
  tflag=(); runner=(); env_prefix=(env)
  if [ "$target" != native ]; then
    tflag=(--target="$target")
    q=$(qemu_for "$target")
    [ -z "$q" ] && { echo "no emulator known for $target"; continue; }
    runner=("$q")
    env_prefix=(env QEMU_LD_PREFIX=/usr/$target WINEDEBUG=-all WINEPREFIX=${WINEPREFIX:-/tmp/pk-wineprefix})
  fi
  dir="$OUT/$target"
  mkdir -p "$dir"
  for t in "${single[@]}" t09_multi bitlearn_tm; do
    args=()
    if [ "$t" = t09_multi ]; then
      srcs=("$PK/tests/t09_multi/kernels.cu" "$PK/tests/t09_multi/main.cu")
    elif [ "$t" = bitlearn_tm ]; then   # CUDA port of bitlearn's kernels, checked against bitlearn's own results
      srcs=("$PK/examples/bitlearn/bitlearn_tm.cu")
      args=("$PK/examples/bitlearn/tm_case")
    else
      srcs=("$PK/tests/$t.cu")
    fi
    exe="$dir/$t"; case "$target" in *mingw*) exe="$dir/$t.exe" ;; esac
    if ! "${PKCC[@]}" "${tflag[@]}" -O2 -o "$exe" "${srcs[@]}" >"$dir/$t.build.log" 2>&1; then
      results+=("$target	$t	cpu	BUILD-FAIL"); fail=$((fail+1)); continue
    fi
    start=$(date +%s.%N)
    if "${env_prefix[@]}" PK_BACKEND=cpu timeout 600 "${runner[@]}" "$exe" "${args[@]}" >"$dir/$t.cpu.log" 2>&1 && grep -q PASS "$dir/$t.cpu.log"; then
      secs=$(awk -v a="$start" -v b="$(date +%s.%N)" 'BEGIN{print b - a}')
      results+=("$target	$t	cpu	PASS	$(printf '%.2fs' "$secs")"); pass=$((pass+1))
    else
      results+=("$target	$t	cpu	FAIL	$(tail -1 "$dir/$t.cpu.log")"); fail=$((fail+1))
    fi
    if [ "$target" = native ] && [ $have_opencl = 1 ]; then
      # "opencl" uses the dialect the device reports; "opencl-1.2" forces OpenCL C 1.2 (what NVIDIA drivers accept)
      for mode in ${PK_OCL_MODES:-opencl opencl-1.2}; do
        std=""; [ $mode = opencl-1.2 ] && std=CL1.2
        log="$dir/$t.$mode.log"
        start=$(date +%s.%N)
        if PK_OPENCL_STD=$std PK_BACKEND=opencl PK_VERBOSE=1 timeout 600 "$exe" "${args[@]}" >"$log" 2>&1 && grep -q PASS "$log"; then
          secs=$(awk -v a="$start" -v b="$(date +%s.%N)" 'BEGIN{print b - a}')
          if grep -q "backend: opencl" "$log"; then
            results+=("$target	$t	$mode	PASS	$(printf '%.2fs' "$secs") $(grep -o 'backend: opencl.*' "$log" | head -1 | sed 's/backend: opencl //')"); pass=$((pass+1))
          elif [[ "$cpu_only" == *" $t "* ]]; then
            results+=("$target	$t	$mode	PASS	CPU fallback (expected: CPU-only features)"); pass=$((pass+1))
          elif [[ "$cpu_only_cl12" == *" $t "* ]] && grep -q "CL1.2" "$log"; then
            results+=("$target	$t	$mode	PASS	CPU fallback (expected on OpenCL C 1.2: program-scope variables)"); pass=$((pass+1))
          else
            results+=("$target	$t	$mode	FAIL	unexpected CPU fallback: $(grep 'backend:' "$log")"); fail=$((fail+1))
          fi
        else
          results+=("$target	$t	$mode	FAIL	$(tail -1 "$log")"); fail=$((fail+1))
        fi
      done
    fi
    if [ "$target" = native ] && [ $have_ptx = 1 ]; then
      log="$dir/$t.ptx.log"
      start=$(date +%s.%N)
      if PK_BACKEND=ptx PK_VERBOSE=1 timeout 900 "$exe" "${args[@]}" >"$log" 2>&1 && grep -q PASS "$log"; then
        secs=$(awk -v a="$start" -v b="$(date +%s.%N)" 'BEGIN{print b - a}')
        if grep -q "backend: ptx" "$log"; then
          results+=("$target	$t	ptx	PASS	$(printf '%.2fs' "$secs") ($(grep -o 'backend: ptx ([^;]*' "$log" | head -1 | sed 's/backend: ptx (//'))"); pass=$((pass+1))
        elif [[ "$cpu_only_ptx" == *" $t "* ]]; then
          results+=("$target	$t	ptx	PASS	CPU fallback (expected: $(grep -o 'backend: cpu .* - .*' "$log" | head -1 | sed 's/^[^-]*- [^:]*: //' | cut -c1-70))"); pass=$((pass+1))
        else
          results+=("$target	$t	ptx	FAIL	unexpected CPU fallback: $(grep 'backend:' "$log")"); fail=$((fail+1))
        fi
      else
        results+=("$target	$t	ptx	FAIL	$(grep -v 'backend:' "$log" | tail -1)"); fail=$((fail+1))
      fi
    fi
  done
done
printf '%b\n' "${results[@]}" | column -t -s $'\t' 2>/dev/null || printf '%b\n' "${results[@]}"
echo "passed: $pass  failed: $fail"
[ $fail -eq 0 ]
