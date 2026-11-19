#!/usr/bin/env python3
"""Build a self-contained Kaggle notebook that tests PolyKernel on an NVIDIA GPU (T4 or P100).

    python3 tools/make_kaggle_notebook.py [polykernel.tar.gz] [out.ipynb]

The source tarball is embedded as base64, so the notebook is the only file to upload.
"""
import base64
import hashlib
import json
import os
import subprocess
import sys

PK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# cuda-samples that build and pass with PolyKernel on the CPU (commit below); the notebook runs them
# on the GPU through PolyKernel's PTX compiler and through OpenCL and, for reference, builds them with nvcc.
SAMPLES_COMMIT = '0ec046d5e99d7214cc8e9fbac2cc1e66f3019878'
SAMPLES = """0_Introduction/asyncAPI 0_Introduction/simpleAtomicIntrinsics 0_Introduction/simpleCallback
0_Introduction/simpleCooperativeGroups 0_Introduction/simpleMultiCopy 0_Introduction/simplePrintf
0_Introduction/simpleStreams 0_Introduction/simpleVoteIntrinsics 0_Introduction/simpleZeroCopy
0_Introduction/systemWideAtomics 0_Introduction/vectorAdd 1_Utilities/deviceQuery
2_Concepts_and_Techniques/convolutionSeparable 2_Concepts_and_Techniques/eigenvalues
2_Concepts_and_Techniques/histogram 2_Concepts_and_Techniques/scalarProd 2_Concepts_and_Techniques/shfl_scan
2_Concepts_and_Techniques/sortingNetworks 2_Concepts_and_Techniques/threadFenceReduction
3_CUDA_Features/StreamPriorities 3_CUDA_Features/binaryPartitionCG 3_CUDA_Features/newdelete
3_CUDA_Features/warpAggregatedAtomicsCG 5_Domain_Specific/BlackScholes 5_Domain_Specific/SobolQRNG
5_Domain_Specific/binomialOptions 5_Domain_Specific/dwtHaar1D 5_Domain_Specific/fastWalshTransform
5_Domain_Specific/quasirandomGenerator 6_Performance/LargeKernelParameter 6_Performance/alignedTypes
6_Performance/transpose""".split()


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(True)}


def code(text, hide=False):
    meta = {'_kg_hide-input': True, 'jupyter': {'source_hidden': True}} if hide else {}
    return {'cell_type': 'code', 'execution_count': None, 'metadata': meta, 'outputs': [],
            'source': text.strip('\n').splitlines(True)}


INTRO = """
# PolyKernel on a Kaggle GPU (T4 or P100)

PolyKernel runs unmodified CUDA source on any CPU or GPU. On NVIDIA GPUs it has its own compiler:
`pkcc` turns the CUDA kernels into PTX (NVIDIA's documented GPU assembly language) and the program
loads that PTX through the NVIDIA driver, so no CUDA toolkit is needed to run it. This notebook checks
that path, the OpenCL path, and NVIDIA's own `nvcc` on the same card.

**Before you run:** in the right-hand panel set **Accelerator → GPU T4 x2** (or P100) and **Internet → On**
(Internet is optional: without it the Python 3.14, clang and cuda-samples steps are skipped).
Then **Run All**. It takes about 20–35 minutes.

What it does:

1. Unpacks PolyKernel (embedded in the next cell) and reports the GPU, driver, CUDA and OpenCL setup.
2. Runs the PolyKernel test suite on the CPU back end, on the GPU through PolyKernel's own PTX compiler,
   and on the GPU through OpenCL.
3. Builds the same test programs with `nvcc` and runs them natively as the reference.
4. Checks the bitlearn Tsetlin Machine port bit for bit (CPU, PTX, OpenCL, nvcc).
5. Times a matrix-multiply benchmark: native CUDA vs PolyKernel PTX vs PolyKernel OpenCL on the same GPU.
6. Runs 32 unmodified NVIDIA cuda-samples through PolyKernel (PTX and OpenCL) and with nvcc (needs Internet).

**At the end:** open the **Output** panel (or `/kaggle/working`) and download
`polykernel_results.txt` (summary) and `polykernel_log.txt` (full logs), and send them back.
The last cell also prints the summary, so copying that text works too.
"""

HELPERS = r'''
# Helpers, unpacking and environment report
import os, sys, io, re, json, time, glob, base64, hashlib, shutil, tarfile, platform, subprocess, traceback

OUT = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('polykernel_out')
WORK = '/tmp/pk'
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
LOG = os.path.join(OUT, 'polykernel_log.txt')
open(LOG, 'w').close()
RES = {'started': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), 'errors': [], 'env': {}}
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')


def log(text):
    with open(LOG, 'a', encoding='utf-8', errors='replace') as f:
        f.write(text if text.endswith('\n') else text + '\n')


def sh(cmd, timeout=1200, env=None, cwd=None, show=25):
    """Run a shell command, log everything, print the tail. Returns (exit code, output)."""
    e = dict(os.environ)
    e.update(env or {})
    t0 = time.time()
    try:
        p = subprocess.run(['bash', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e, cwd=cwd,
                           timeout=timeout, text=True, encoding='utf-8', errors='replace')
        rc, out = p.returncode, p.stdout
    except subprocess.TimeoutExpired as ex:
        out = ex.stdout if isinstance(ex.stdout, str) else (ex.stdout or b'').decode('utf-8', 'replace')
        rc, out = 124, (out or '') + '\n[timed out after %ds]' % timeout
    log('\n$ %s%s\n%s\n[exit %d, %.1fs]' % (cmd, ('   (in %s)' % cwd) if cwd else '', out.rstrip(), rc, time.time() - t0))
    if show:
        lines = out.rstrip().splitlines()
        print('\n'.join(lines[-show:]))
    return rc, out


class Step:
    """Times a block, logs it, and keeps going if it fails."""
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time()
        print('=' * 20, self.name)
        log('\n' + '#' * 30 + ' ' + self.name)
        return self
    def __exit__(self, et, ev, tb):
        dt = time.time() - self.t0
        if et is not None:
            msg = '%s: %s: %s' % (self.name, et.__name__, ev)
            RES['errors'].append(msg)
            log(traceback.format_exc())
            print('STEP FAILED (continuing):', msg)
        print('-- %s took %.0fs' % (self.name, dt))
        return True


def first(cmd):
    rc, out = sh(cmd, show=0)
    return out.strip().splitlines()[0].strip() if rc == 0 and out.strip() else ''


def internet():
    import urllib.request
    try:
        urllib.request.urlopen('https://pypi.org/simple/pip/', timeout=8)
        return True
    except Exception:
        return False


with Step('unpack PolyKernel'):
    data = base64.b64decode(''.join(PK_TARBALL_B64.split()))
    assert hashlib.sha256(data).hexdigest() == PK_TARBALL_SHA256, 'embedded tarball is corrupt'
    shutil.rmtree(os.path.join(WORK, 'polykernel'), ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(WORK, filter='data') if hasattr(tarfile, 'data_filter') else t.extractall(WORK)
    PK = os.path.join(WORK, 'polykernel')
    os.chmod(os.path.join(PK, 'bin', 'pkcc'), 0o755)
    os.chmod(os.path.join(PK, 'tests', 'run_tests.sh'), 0o755)
    RES['polykernel'] = first('python3 %s/bin/pkcc --version' % PK) or 'unknown'
    print('PolyKernel unpacked to', PK, '-', RES['polykernel'])

with Step('environment'):
    env = RES['env']
    env['os'] = first("grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '\"'")
    env['cpu'] = first("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2") + ' x%d' % (os.cpu_count() or 0)
    env['gpu'] = first('nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader') \
        or first('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader') or 'none found'
    m = re.search(r'CUDA Version: ([\d.]+)', sh('nvidia-smi', show=0)[1])
    env['driver_cuda'] = m.group(1) if m else ''
    NVCC = shutil.which('nvcc') or ''
    env['nvcc'] = (re.findall(r'release ([\d.]+)', sh(NVCC + ' --version', show=0)[1]) or ['?'])[0] if NVCC else 'not found'
    env['cxx'] = first('g++ --version') or first('c++ --version')
    env['python_system'] = sys.version.split()[0]
    env['internet'] = internet()
    rc, libs = sh("ldconfig -p | grep -E 'libOpenCL|libnvidia-opencl' ; ls /etc/OpenCL/vendors 2>/dev/null; "
                  "ls /usr/local/nvidia/lib64/libnvidia-opencl* /usr/lib/x86_64-linux-gnu/libnvidia-opencl* 2>/dev/null",
                  show=0)
    env['opencl_libs'] = ' | '.join(sorted(set(l.strip() for l in libs.splitlines() if l.strip())))[:400]
    rc, libs = sh("ldconfig -p | grep -E 'libcuda\\.so'; for d in ${LD_LIBRARY_PATH//:/ } /usr/local/nvidia/lib64 "
                  "/usr/lib/x86_64-linux-gnu /usr/lib64; do ls $d/libcuda.so* 2>/dev/null; done", show=0)
    env['libcuda'] = ' | '.join(sorted(set(l.strip() for l in libs.splitlines() if l.strip())))[:300] or 'not found'
    env['ptxas'] = shutil.which('ptxas') or ('/usr/local/cuda/bin/ptxas' if os.path.exists('/usr/local/cuda/bin/ptxas') else
                                             'none (PTX is checked by the driver only)')
    cc = env['gpu'].split(',')[-1].strip() if env['gpu'].count(',') >= 3 else ''
    ARCH = 'sm_' + cc.replace('.', '') if re.match(r'^\d+\.\d+$', cc) else 'sm_60'
    env['nvcc_arch'] = ARCH
    for k, v in env.items():
        print('%-14s %s' % (k, v))
'''

OPENCL = r'''
# OpenCL: the NVIDIA driver ships the OpenCL implementation (libnvidia-opencl.so.1).
# PolyKernel uses the standard ICD loader if present, and can also talk to the driver directly.
with Step('OpenCL setup'):
    vend = ''
    for d in ['/usr/local/nvidia/lib64', '/usr/lib/x86_64-linux-gnu', '/usr/lib64', '/usr/local/cuda/compat']:
        c = glob.glob(os.path.join(d, 'libnvidia-opencl.so*'))
        if c:
            vend = sorted(c, key=len)[0]
            break
    RES['env']['nvidia_opencl'] = vend or 'not found'
    if vend:
        os.environ['PK_OPENCL_VENDOR_LIBRARY'] = vend   # used only when there is no libOpenCL
        try:
            os.makedirs('/etc/OpenCL/vendors', exist_ok=True)
            if not any('nvidia' in open(f).read() for f in glob.glob('/etc/OpenCL/vendors/*.icd')):
                with open('/etc/OpenCL/vendors/nvidia.icd', 'w') as f:
                    f.write(vend + '\n')
                print('wrote /etc/OpenCL/vendors/nvidia.icd ->', vend)
        except OSError as ex:
            print('could not write the ICD file (%s); PolyKernel will load the driver directly' % ex)
    else:
        print('WARNING: libnvidia-opencl.so not found; the OpenCL runs will fall back to the CPU')
    have_loader = 'libOpenCL.so' in sh('ldconfig -p', show=0)[1]
    if RES['env']['internet']:
        pk = ['clinfo'] + ([] if have_loader else ['ocl-icd-libopencl1']) + ([] if shutil.which('clang') else ['clang'])
        sh('apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ' + ' '.join(pk), timeout=900, show=5)
    else:
        print('no Internet: skipping clinfo/clang; PolyKernel uses the driver directly if there is no ICD loader')
    RES['env']['icd_loader'] = 'yes' if 'libOpenCL.so' in sh('ldconfig -p', show=0)[1] else 'no (PolyKernel built-in ICD path)'
    RES['env']['clang'] = first('clang --version') or 'none (OpenCL output is not pre-validated)'
    if shutil.which('clinfo'):
        rc, out = sh("clinfo | grep -E 'Platform Name|Device Name|Device Version|OpenCL C Version|Driver Version' | head -12")
        RES['env']['clinfo'] = ' | '.join(l.strip() for l in out.splitlines())[:600]
'''

PYTHON = r'''
# Python 3.14 for PolyKernel's compiler driver (falls back to the system Python)
with Step('Python 3.14'):
    PY = sys.executable
    if RES['env']['internet']:
        sh('pip install -q uv', timeout=300, show=3)
        rc, out = sh('uv python install 3.14 >/dev/null 2>&1; uv python find 3.14', timeout=600, show=3)
        cand = out.strip().splitlines()[-1].strip() if out.strip() else ''
        if rc == 0 and os.path.exists(cand):
            PY = cand
    os.environ['PK_PYTHON'] = PY
    RES['env']['python_pkcc'] = first(PY + ' --version') + ' (' + PY + ')'
    print('pkcc runs with', RES['env']['python_pkcc'])
'''

TESTS = r'''
# PolyKernel test suite: every test on the CPU back end, then on the GPU through PolyKernel's own PTX
# compiler (NVIDIA driver API) and through OpenCL
def parse_table(text):
    rows = []
    for line in text.splitlines():
        m = re.match(r'^(native)\s+(\S+)\s+(cpu|opencl\S*|ptx)\s+(PASS|FAIL)\s*(.*)$', line.strip())
        if m:
            rows.append({'test': m.group(2), 'mode': m.group(3), 'status': m.group(4), 'note': m.group(5).strip()})
    return rows

with Step('PolyKernel tests'):
    rc, out = sh('tests/run_tests.sh native', cwd=PK, timeout=3600, show=40,
                 env={'PK_TEST_OUT': WORK + '/tests', 'PK_OCL_MODES': 'opencl'})
    RES['tests'] = parse_table(out)
    for r in RES['tests']:   # attach logs of failures
        if r['status'] != 'PASS':
            for f in glob.glob(WORK + '/tests/**/%s*.log' % r['test'], recursive=True):
                log('\n----- %s\n%s' % (f, open(f, encoding='utf-8', errors='replace').read()[-6000:]))
    for mode in ('ptx', 'opencl'):
        gpu_rows = [r for r in RES['tests'] if r['mode'].startswith(mode) and 'CPU fallback' not in r['note']]
        print('ran on %s device:' % mode, gpu_rows[0]['note'] if gpu_rows else 'none')
    # verbose runs, for the record (device, PTX JIT log, OpenCL dialect)
    rc, out = sh('PK_BACKEND=ptx PK_VERBOSE=1 %s/native/t01_vector_add' % (WORK + '/tests'), show=8)
    rc, out = sh('PK_BACKEND=opencl PK_VERBOSE=1 %s/native/t01_vector_add' % (WORK + '/tests'), show=8)
    # the PTX PolyKernel generated for t01 (start of the vecAdd kernel), checked with NVIDIA's assembler if present
    rc, out = sh('$PK_PYTHON bin/pkcc --keep -O2 -c -o %s/t01.o tests/t01_vector_add.cu && '
                 'grep -n -A24 "entry pk_k_vecAdd" %s/t01.ptx | head -30' % (WORK, WORK), cwd=PK, show=32)
    if os.path.exists(RES['env'].get('ptxas', '')):
        sh('%s -arch=%s -v %s/t01.ptx -o /dev/null' % (RES['env']['ptxas'], ARCH, WORK), show=8)
'''

NVCC = r'''
# Reference: the same programs built with NVIDIA's nvcc and run natively
NV_PROGS = [(t, ['tests/%s.cu' % t], []) for t in
            ['t01_vector_add', 't02_matmul_tiled', 't03_reduction', 't04_histogram', 't05_constant_struct',
             't06_runtime_api', 't07_templates_lambdas', 't08_warp_coop', 't10_opencl_features',
             't11_coalesced_groups', 't12_device_vars', 't13_cg_partitions']]
NV_PROGS += [('t09_multi', ['tests/t09_multi/kernels.cu', 'tests/t09_multi/main.cu'], []),
             ('bitlearn_tm', ['examples/bitlearn/bitlearn_tm.cu'], ['examples/bitlearn/tm_case'])]
RES['nvcc'] = []
with Step('nvcc reference'):
    if not NVCC:
        raise RuntimeError('nvcc not found')
    os.makedirs(WORK + '/nvcc', exist_ok=True)
    flags = '-O2 -arch=%s -Wno-deprecated-gpu-targets --extended-lambda -std=c++17 -Itests -w' % ARCH
    for name, srcs, args in NV_PROGS:
        exe = WORK + '/nvcc/' + name
        rc, out = sh('nvcc %s -o %s %s' % (flags, exe, ' '.join(srcs)), cwd=PK, timeout=900, show=0)
        if rc != 0:
            why = 'build failed'
            if 'Unsupported gpu architecture' in out or 'not supported' in out and ARCH in out:
                why = 'this nvcc cannot target %s' % ARCH
            elif name in ('t11_coalesced_groups', 't13_cg_partitions') and ('labeled_partition' in out or 'match' in out):
                why = 'needs compute capability 7.0+ (labeled_partition/match), not available on this GPU'
            RES['nvcc'].append({'test': name, 'status': 'n/a', 'note': why})
            print('%-22s %s' % (name, why))
            continue
        t0 = time.time()
        rc, out = sh('%s %s' % (exe, ' '.join(args)), cwd=PK, timeout=600, show=0)
        ok = rc == 0 and 'PASS' in out
        line = ([l for l in out.splitlines() if 'PASS' in l or 'FAIL' in l] or [out.strip()[-120:]])[-1]
        RES['nvcc'].append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'note': '%.2fs %s' % (time.time() - t0, line[:120])})
        print('%-22s %s  %s' % (name, 'PASS' if ok else 'FAIL', line[:100]))
'''

BITLEARN = r'''
# bitlearn Tsetlin Machine port: one epoch of training + inference must match bitlearn (Numba) bit for bit
with Step('bitlearn check'):
    RES['bitlearn'] = {}
    for mode, env in [('PolyKernel CPU', {'PK_BACKEND': 'cpu', 'PK_VERBOSE': '1'}),
                      ('PolyKernel PTX (GPU)', {'PK_BACKEND': 'ptx', 'PK_VERBOSE': '1'}),
                      ('PolyKernel OpenCL (GPU)', {'PK_BACKEND': 'opencl', 'PK_VERBOSE': '1'})]:
        rc, out = sh('%s/native/bitlearn_tm examples/bitlearn/tm_case' % (WORK + '/tests'), cwd=PK, env=env, show=0)
        be = (re.findall(r'backend: (\S+)', out) or ['?'])[0]
        RES['bitlearn'][mode] = ('PASS' if rc == 0 and 'PASS' in out else 'FAIL') + ' (backend %s)' % be
    nv = [r for r in RES.get('nvcc', []) if r['test'] == 'bitlearn_tm']
    RES['bitlearn']['nvcc native (GPU)'] = nv[0]['status'] if nv else 'not run'
    for k, v in RES['bitlearn'].items():
        print('%-26s %s' % (k, v))
'''

BENCH = r'''
# SGEMM benchmark: the same source built with nvcc (native CUDA) and with pkcc (PolyKernel's own PTX, and OpenCL)
def bench_rows(out, backend):
    rows = []
    for m in re.finditer(r'RESULT kernel=(\w+) N=(\d+) ms=([\d.]+) gflops=([\d.]+) maxrelerr=(\S+) (OK|WRONG)', out):
        rows.append({'backend': backend, 'kernel': m.group(1), 'N': int(m.group(2)), 'ms': float(m.group(3)),
                     'gflops': float(m.group(4)), 'check': m.group(6)})
    return rows

RES['bench'] = []
with Step('SGEMM benchmark'):
    rc, _ = sh('$PK_PYTHON bin/pkcc -O2 -o %s/bench_pk examples/bench_sgemm.cu' % WORK, cwd=PK, timeout=900, show=5)
    if NVCC:
        sh('nvcc -O3 -arch=%s -Wno-deprecated-gpu-targets -o %s/bench_nv examples/bench_sgemm.cu' % (ARCH, WORK), cwd=PK,
           timeout=900, show=5)
    for N, reps in [(1024, 20), (2048, 10), (4096, 5)]:
        if os.path.exists(WORK + '/bench_nv'):
            RES['bench'] += bench_rows(sh('%s/bench_nv %d %d' % (WORK, N, reps), timeout=900, show=4)[1], 'CUDA (nvcc)')
        slow = False
        for be, label in [('ptx', 'PolyKernel PTX'), ('opencl', 'PolyKernel OpenCL')]:
            t0 = time.time()
            out = sh('%s/bench_pk %d %d' % (WORK, N, reps), env={'PK_BACKEND': be, 'PK_VERBOSE': '1'}, timeout=900, show=5)[1]
            if ('backend: ' + be) in out:
                RES['bench'] += bench_rows(out, label)
            else:
                print('%s: not on the GPU (CPU fallback), not counted' % label)
            slow = slow or time.time() - t0 > 60
        if slow:   # not a fast GPU: skip the larger sizes
            print('a run took over a minute; skipping larger sizes')
            break
    RES['bench'] += bench_rows(sh('%s/bench_pk 1024 1' % WORK, env={'PK_BACKEND': 'cpu'}, timeout=1200, show=4)[1],
                               'PolyKernel CPU')
'''

SAMPLES_CELL = r'''
# Unmodified NVIDIA cuda-samples through PolyKernel on the GPU (and with nvcc for reference). Needs Internet.
SAMPLES_COMMIT = '%s'
SAMPLES = %r
RES['samples'] = []
with Step('cuda-samples'):
    if not RES['env']['internet']:
        raise RuntimeError('no Internet, skipped')
    CS = WORK + '/cuda-samples'
    if not os.path.isdir(CS):
        sh('git clone -q --filter=blob:none https://github.com/NVIDIA/cuda-samples %%s && cd %%s && git checkout -q %%s'
           %% (CS, CS, SAMPLES_COMMIT), timeout=900, show=3)
    os.makedirs(WORK + '/samples', exist_ok=True)
    for s in SAMPLES:
        d = os.path.join(CS, 'cpp', s)
        if not os.path.isdir(d):
            d = os.path.join(CS, 'Samples', s)
        name = s.split('/')[-1]
        srcs = ' '.join(sorted(glob.glob(d + '/*.cu') + glob.glob(d + '/*.cpp')))
        row = {'sample': s}
        exe = WORK + '/samples/' + name
        rc, out = sh('$PK_PYTHON %%s/bin/pkcc -O2 -I%%s/Common -I%%s -o %%s %%s' %% (PK, CS, d, exe, srcs), timeout=600, show=0)
        if rc != 0:
            row['ptx'] = row['opencl'] = 'build failed'
        else:
            for be in ('ptx', 'opencl'):
                rc, out = sh(exe, cwd=d, env={'PK_BACKEND': be, 'PK_VERBOSE': '1'}, timeout=600, show=0)
                on = ('backend: ' + be) in out
                row[be] = ('PASS' if rc == 0 else 'FAIL(%%d)' %% rc) + ' on ' + ('GPU' if on else 'CPU fallback')
        if NVCC:
            nexe = exe + '_nv'
            rc, out = sh('nvcc -O2 -arch=%%s -Wno-deprecated-gpu-targets -w -I%%s/Common -I%%s -o %%s %%s'
                         %% (ARCH, CS, d, nexe, srcs), timeout=600, show=0)
            if rc != 0:
                err = [l for l in out.splitlines() if 'error' in l]
                row['nvcc'] = 'build failed: ' + (err[0].split('error')[-1].strip(' :')[:70] if err else '?')
            else:
                rc, out = sh(nexe, cwd=d, timeout=600, show=0)
                row['nvcc'] = 'PASS' if rc == 0 else 'FAIL(%%d)' %% rc
        RES['samples'].append(row)
        print('%%-42s PTX: %%-22s OpenCL: %%-22s nvcc: %%s' %% (s, row['ptx'], row['opencl'], row.get('nvcc', '-')))
''' % (SAMPLES_COMMIT, SAMPLES)

SUMMARY = r'''
# Summary -> polykernel_results.txt (send this back, together with polykernel_log.txt)
def summary():
    L = ['PolyKernel GPU test results', '=' * 60, 'started  %s' % RES['started'],
         'finished %s' % time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), RES.get('polykernel', ''), '']
    L += ['%-14s %s' % (k, v) for k, v in RES['env'].items()]
    t = RES.get('tests', [])
    if t:
        L += ['', 'PolyKernel test suite (%d/%d passed)' % (sum(r['status'] == 'PASS' for r in t), len(t))]
        L += ['  %-22s %-7s %-5s %s' % (r['test'], r['mode'], r['status'], r['note']) for r in t]
    if RES.get('nvcc'):
        n = RES['nvcc']
        L += ['', 'nvcc reference on the same GPU (%d passed, %d not applicable)' % (
            sum(r['status'] == 'PASS' for r in n), sum(r['status'] == 'n/a' for r in n))]
        L += ['  %-22s %-5s %s' % (r['test'], r['status'], r['note']) for r in n]
    if RES.get('bitlearn'):
        L += ['', 'bitlearn Tsetlin Machine port (bit-exact vs bitlearn/Numba)']
        L += ['  %-26s %s' % kv for kv in RES['bitlearn'].items()]
    if RES.get('bench'):
        L += ['', 'SGEMM benchmark (GFLOP/s, higher is better)',
              '  %-18s %-6s %6s %10s %10s %s' % ('backend', 'kernel', 'N', 'ms', 'GFLOP/s', 'check')]
        L += ['  %-18s %-6s %6d %10.3f %10.1f %s' % (r['backend'], r['kernel'], r['N'], r['ms'], r['gflops'], r['check'])
              for r in RES['bench']]
        nat = {(r['kernel'], r['N']): r['gflops'] for r in RES['bench'] if r['backend'] == 'CUDA (nvcc)'}
        for be in ('PolyKernel PTX', 'PolyKernel OpenCL'):
            for r in RES['bench']:
                if r['backend'] == be and nat.get((r['kernel'], r['N'])):
                    L.append('  %-17s / CUDA %-6s N=%-5d %5.0f%%' % (be, r['kernel'], r['N'],
                                                               100 * r['gflops'] / nat[(r['kernel'], r['N'])]))
    if RES.get('samples'):
        s = RES['samples']
        L += ['', 'cuda-samples (%d): nvcc %d pass' % (len(s), sum(r.get('nvcc') == 'PASS' for r in s))]
        for be in ('ptx', 'opencl'):
            L.append('  PolyKernel %-6s %d pass on GPU, %d pass with CPU fallback, %d failed/not built' % (
                be, sum(r[be] == 'PASS on GPU' for r in s), sum(r[be] == 'PASS on CPU fallback' for r in s),
                sum(not r[be].startswith('PASS') for r in s)))
        L += ['  %-42s PTX: %-22s OpenCL: %-22s nvcc: %s' % (r['sample'], r['ptx'], r['opencl'], r.get('nvcc', '-'))
              for r in s]
    L += ['', 'errors: %d' % len(RES['errors'])] + ['  ' + e for e in RES['errors']]
    return '\n'.join(L) + '\n'

text = summary()
with open(os.path.join(OUT, 'polykernel_results.txt'), 'w', encoding='utf-8') as f:
    f.write(text)
with open(os.path.join(OUT, 'polykernel_results.json'), 'w', encoding='utf-8') as f:
    json.dump(RES, f, indent=1)
print(text)
print('Saved to', OUT, ':', ', '.join(sorted(os.listdir(OUT))))
'''


def main():
    tar = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PK, '..', 'polykernel_kaggle_gpu_test.ipynb')
    if not tar:
        tar = subprocess.run(['bash', os.path.join(PK, 'tools', 'make_dist.sh'), '/tmp/polykernel-kaggle.tar.gz'],
                             check=True, stdout=subprocess.PIPE, text=True).stdout.strip().splitlines()[-1]
    with open(tar, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode('ascii')
    lines = [b64[i:i + 1000] for i in range(0, len(b64), 1000)]
    payload = ('# PolyKernel source code (embedded tarball, %d KB). Nothing to edit here.\n'
               'PK_TARBALL_SHA256 = %r\nPK_TARBALL_B64 = """\n%s\n"""\nprint("payload loaded:", len(PK_TARBALL_B64) // 1024, "KB")'
               % (len(data) // 1024, hashlib.sha256(data).hexdigest(), '\n'.join(lines)))
    cells = [md(INTRO), code(payload, hide=True), code(HELPERS), code(OPENCL), code(PYTHON), code(TESTS),
             code(NVCC), code(BITLEARN), code(BENCH), code(SAMPLES_CELL), code(SUMMARY)]
    nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'},
                       'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': True, 'isGpuEnabled': True}}}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(os.path.abspath(out), '%d KB' % (os.path.getsize(out) // 1024))


if __name__ == '__main__':
    main()
