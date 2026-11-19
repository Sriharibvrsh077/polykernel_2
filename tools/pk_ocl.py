"""Device pass: translate CUDA kernels (C-style subset of CUDA C++) to OpenCL C.

Input is the device-pass preprocessed source (user code only; system headers are
stubbed). Output: OpenCL C source plus metadata (kernel signatures, symbols).
Anything outside the supported subset (templates, classes with methods, warp
intrinsics, references, ...) is skipped; kernels that depend on skipped items are
dropped too, and launches of dropped kernels make the program use the CPU back end.
"""
import os
import re
import shutil
import subprocess
import tempfile

from pk_lex import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
IDENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


class Unsupported(Exception):
    pass


SYNC_IDS = {
    '__syncthreads', '__syncthreads_count', '__syncthreads_and', '__syncthreads_or', '__syncwarp',
    '__shfl_sync', '__shfl_up_sync', '__shfl_down_sync', '__shfl_xor_sync', '__shfl', '__shfl_up',
    '__shfl_down', '__shfl_xor', '__ballot_sync', '__ballot', '__any_sync', '__all_sync', '__any',
    '__all', '__uni_sync', '__match_any_sync', '__match_all_sync', '__reduce_add_sync',
    '__reduce_min_sync', '__reduce_max_sync', '__reduce_and_sync', '__reduce_or_sync',
    '__reduce_xor_sync', 'cooperative_groups', 'this_thread_block', 'tiled_partition',
    'coalesced_threads', 'this_grid', 'thread_block', 'thread_block_tile', 'barrier_wait',
}

UNSUPPORTED_IDS = {
    'template', 'typename', 'new', 'delete', 'this', 'throw', 'try', 'catch', 'operator', 'auto',
    'decltype', 'virtual', 'dynamic_cast', 'typeid', 'asm', '__asm__', '__asm', 'malloc', 'free',
    'goto', 'thrust', 'cub', 'mutable', 'friend', 'explicit', 'namespace', 'using', 'class',
    '__syncthreads_count', '__syncthreads_and', '__syncthreads_or', '__syncwarp', '__activemask',
    '__shfl_sync', '__shfl_up_sync', '__shfl_down_sync', '__shfl_xor_sync', '__shfl', '__shfl_up',
    '__shfl_down', '__shfl_xor', '__ballot_sync', '__ballot', '__any_sync', '__all_sync', '__any',
    '__all', '__uni_sync', '__match_any_sync', '__match_all_sync', '__reduce_add_sync',
    '__reduce_min_sync', '__reduce_max_sync', '__reduce_and_sync', '__reduce_or_sync',
    '__reduce_xor_sync', 'cooperative_groups', 'tex1Dfetch', 'tex1D', 'tex2D', 'tex3D', 'tex1DLayered',
    'tex2DLayered', 'texCubemap', 'surf1Dread', 'surf1Dwrite', 'surf2Dread', 'surf2Dwrite',
    '__half', '__half2', '__nv_bfloat16', 'half2', '__float2half', '__half2float',
    'atomicAdd_block', 'atomicAdd_system', 'static_assert', 'alignof', 'nullptr_t',
    'tiled_partition', 'thread_block_tile', 'coalesced_threads', 'coalesced_group', 'this_grid', 'grid_group',
    'thread_group', 'block_tile_memory', 'memcpy_async', 'labeled_partition', 'binary_partition',
}
# block-level cooperative-groups members -> OpenCL work-group built-ins
CG_MEMBERS = {
    'sync': 'barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)',
    'barrier_wait': 'barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)',
    'thread_rank': 'pk_block_rank()',
    'size': 'pk_block_size()',
    'num_threads': 'pk_block_size()',
    'group_index': 'pk_make_uint3((uint)get_group_id(0), (uint)get_group_id(1), (uint)get_group_id(2))',
    'thread_index': 'pk_make_uint3((uint)get_local_id(0), (uint)get_local_id(1), (uint)get_local_id(2))',
    'group_dim': 'pk_make_uint3((uint)get_local_size(0), (uint)get_local_size(1), (uint)get_local_size(2))',
    'dim_threads': 'pk_make_uint3((uint)get_local_size(0), (uint)get_local_size(1), (uint)get_local_size(2))',
}

IDX_FN = {'threadIdx': 'get_local_id', 'blockIdx': 'get_group_id', 'blockDim': 'get_local_size',
          'gridDim': 'get_num_groups'}
CXX_CASTS = {'static_cast', 'reinterpret_cast', 'const_cast'}
DROP_SPEC = {'__global__', '__device__', '__host__', '__forceinline__', '__noinline__', 'inline',
             '__inline__', '__inline', 'static', 'extern', 'constexpr', '__launch_bounds__',
             '__grid_constant__', 'register'}
ATTR_FUNCS = {'__attribute__', '__launch_bounds__', 'alignas', '__declspec'}
FUNC_CAST_TYPES = {'float', 'double', 'int', 'unsigned', 'char', 'short', 'long', 'bool', 'uint',
                   'size_t', 'uchar', 'ushort', 'ulong', 'int8_t', 'uint8_t', 'int16_t', 'uint16_t',
                   'int32_t', 'uint32_t', 'int64_t', 'uint64_t'}
CAST_PREV = {'(', ',', '=', 'return', '[', '?', ':', '+', '-', '*', '/', '%', '<', '>', '<=', '>=',
             '==', '!=', '&&', '||', '!', '~', '&', '|', '^', '+=', '-=', '*=', '/=', '%=', '{', ';',
             '<<', '>>', '&=', '|=', '^=', '<<=', '>>='}
TYPE_WORDS = {'void', 'char', 'short', 'int', 'long', 'float', 'double', 'bool', 'unsigned', 'signed',
              'const', 'volatile', 'struct', 'enum', 'union', '__restrict__', '__restrict', 'restrict',
              'size_t', 'uint', 'uchar', 'ushort', 'ulong', '*'}

RENAMES = {}
for _b in ('char', 'uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong', 'float', 'double'):
    RENAMES[_b + '1'] = 'pk_' + _b + '1'
    RENAMES[_b + '3'] = 'pk_' + _b + '3'
    RENAMES['make_' + _b + '1'] = 'pk_make_' + _b + '1'
    RENAMES['make_' + _b + '3'] = 'pk_make_' + _b + '3'
RENAMES.update({'longlong1': 'pk_long1', 'longlong3': 'pk_long3', 'ulonglong1': 'pk_ulong1',
                'ulonglong3': 'pk_ulong3', 'make_longlong3': 'pk_make_long3',
                'make_ulonglong3': 'pk_make_ulong3', 'longlong2': 'long2', 'longlong4': 'long4',
                'ulonglong2': 'ulong2', 'ulonglong4': 'ulong4', 'dim3': 'pk_uint3', '__restrict__': 'restrict',
                '__restrict': 'restrict', 'warpSize': '32', 'nullptr': '0', 'constexpr': 'const',
                '__forceinline__': '', '__noinline__': '', '__host__': '', '__device__': '', 'inline': '',
                '__inline__': '', 'register': ''})

# identifiers that are legal in CUDA C++ but reserved in OpenCL C
OCL_RESERVED = {'local', 'global', 'constant', 'private', 'kernel', 'read_only', 'write_only', 'read_write',
                'uniform', 'pipe', 'half', 'quad', 'complex', 'imaginary', 'vec_step', 'image1d_t',
                'image2d_t', 'image3d_t', 'image1d_array_t', 'image2d_array_t', 'image1d_buffer_t',
                'sampler_t', 'event_t', 'queue_t', 'ndrange_t', 'clk_event_t', 'reserve_id_t', 'generic',
                'program', 'kernel_enqueue_flags_t', 'cl_mem_fence_flags', 'memory_order', 'memory_scope',
                'atomic_int', 'atomic_uint', 'atomic_long', 'atomic_ulong', 'atomic_float', 'atomic_double',
                'atomic_flag', 'uchar2', 'barrier', 'mem_fence', 'get_local_id', 'get_group_id',
                'get_global_id', 'get_local_size', 'get_num_groups', 'get_global_size', 'get_work_dim'}

_INT_TYPE_NAMES = {'size_t', 'ptrdiff_t', 'ssize_t', 'intptr_t', 'uintptr_t', 'int8_t', 'uint8_t', 'int16_t',
                   'uint16_t', 'int32_t', 'uint32_t', 'int64_t', 'uint64_t', 'uint', 'uchar', 'ushort', 'ulong'}
_INT = {
    ('char',): ('char', 'I1'), ('char', 'signed'): ('char', 'I1'), ('char', 'unsigned'): ('uchar', 'I1'),
    ('short',): ('short', 'I2'), ('short', 'unsigned'): ('ushort', 'I2'), ('int',): ('int', 'I4'),
    ('signed',): ('int', 'I4'), ('unsigned',): ('uint', 'I4'), ('long',): ('long', 'I8'),
    ('long', 'unsigned'): ('ulong', 'I8'), ('long', 'long'): ('long', 'I8'),
    ('long', 'long', 'unsigned'): ('ulong', 'I8'), ('bool',): ('uchar', 'I1'), ('size_t',): ('ulong', 'I8'),
    ('ptrdiff_t',): ('long', 'I8'), ('ssize_t',): ('long', 'I8'), ('intptr_t',): ('long', 'I8'),
    ('uintptr_t',): ('ulong', 'I8'), ('int8_t',): ('char', 'I1'), ('uint8_t',): ('uchar', 'I1'),
    ('int16_t',): ('short', 'I2'), ('uint16_t',): ('ushort', 'I2'), ('int32_t',): ('int', 'I4'),
    ('uint32_t',): ('uint', 'I4'), ('int64_t',): ('long', 'I8'), ('uint64_t',): ('ulong', 'I8'),
    ('uint',): ('uint', 'I4'), ('uchar',): ('uchar', 'I1'), ('ushort',): ('ushort', 'I2'),
    ('ulong',): ('ulong', 'I8'),
}
_FLT = {('float',): ('float', 'F4'), ('double',): ('double', 'F8')}


def _canon(words):
    w = [x for x in words if x not in ('const', 'volatile', '__restrict__', '__restrict', 'restrict')]
    if 'signed' in w and 'char' not in w:
        w.remove('signed')
        if not w:
            w = ['int']
    if 'int' in w and len(w) > 1:
        w.remove('int')
    return tuple(sorted(w))


class Ctx:
    def __init__(self):
        self.refparams = {}   # devfunc name -> {arity: set(indices of reference parameters)}
        self.deref = set()    # reference parameters of the function being converted
        self.devfuncs = set()
        self.typedefs = {}    # name -> list of type words
        self.enums = set()
        self.structs = set()
        self.symbols = {}     # __device__/__constant__ variable -> (rewritten base type, dims text)
        self.sym_uses = {}    # function name -> sorted symbol names it uses (directly or via callees)
        self.in_body = False  # rewriting a function body (symbol names are replaced there only)


# ---------------------------------------------------------------------------
# Splitting into top-level items
# ---------------------------------------------------------------------------
class Item:
    def __init__(self, toks, ns):
        self.toks = toks          # [[kind, text], ...] without whitespace (pp lines kept)
        self.ns = list(ns)
        self.kind = 'other'
        self.name = None
        self.defines = set()
        self.refs = set()
        self.cl = None
        self.error = None
        self.sig = None
        self.symbol = None
        self.included = False

    def texts(self):
        return [t for k, t in self.toks if k != 'pp']


def split_items(toks):
    items = []
    ns = []
    cur = []
    depth = pdepth = 0
    i, n = 0, len(toks)
    T = lambda j: toks[j][1] if j < n else ''
    while i < n:
        k, t = toks[i]
        if k == 'pp':
            if depth > 0:
                cur.append(toks[i])
            i += 1
            continue
        if depth == 0 and pdepth == 0 and not cur:
            if t == 'namespace' or (t == 'inline' and T(i + 1) == 'namespace'):
                j = i + (2 if t == 'inline' else 1)
                parts = []
                while j < n and T(j) not in ('{', ';', '='):
                    parts.append(T(j))
                    j += 1
                if T(j) == '{':
                    name = ''.join(parts)
                    ns.append([x for x in name.split('::') if x] if name else ['(anon)'])
                    i = j + 1
                    continue
                while j < n and T(j) != ';':
                    j += 1
                i = j + 1
                continue
            if t == 'extern' and i + 2 < n and toks[i + 1][0] == 'str' and T(i + 2) == '{':
                ns.append([])
                i += 3
                continue
            if t == '}' and ns:
                ns.pop()
                i += 1
                continue
            if t == ';':
                i += 1
                continue
        cur.append(toks[i])
        if t == '(':
            pdepth += 1
        elif t == ')':
            pdepth -= 1
        elif t == '{':
            depth += 1
        elif t == '}':
            depth -= 1
            if depth == 0 and pdepth == 0 and not _needs_semicolon(cur):
                items.append(Item(cur, [x for part in ns for x in part]))
                cur = []
        elif t == ';' and depth == 0 and pdepth == 0:
            items.append(Item(cur, [x for part in ns for x in part]))
            cur = []
        i += 1
    if cur:
        items.append(Item(cur, [x for part in ns for x in part]))
    return items


def _needs_semicolon(cur):
    head = []
    for k, t in cur:
        if t == '{':
            break
        head.append(t)
    d = 0
    has_paren = False
    for t in head:
        if t == '(':
            d += 1
            has_paren = True
        elif t == ')':
            d -= 1
        elif t == '=' and d == 0:
            return True
    if has_paren:
        return False
    return True


# ---------------------------------------------------------------------------
# Helpers on token-text lists
# ---------------------------------------------------------------------------
def _match(tl, i, o='(', c=')'):
    d = 0
    for j in range(i, len(tl)):
        if tl[j] == o:
            d += 1
        elif tl[j] == c:
            d -= 1
            if d == 0:
                return j
    raise Unsupported('unbalanced %s' % o)


def _match_angle(tl, i):
    d = 0
    for j in range(i, len(tl)):
        t = tl[j]
        if t == '<':
            d += 1
        elif t == '>':
            d -= 1
        elif t == '>>':
            d -= 2
        if d <= 0 and t in ('>', '>>'):
            return j
    raise Unsupported('unbalanced <')


def _skip_attrs(tl, j):
    while j < len(tl):
        if tl[j] in ATTR_FUNCS and j + 1 < len(tl) and tl[j + 1] == '(':
            j = _match(tl, j + 1) + 1
        elif tl[j] == '[' and j + 1 < len(tl) and tl[j + 1] == '[':
            j = _match(tl, j, '[', ']') + 1
        else:
            break
    return j


def _strip_attrs(tl):
    out = []
    j = 0
    while j < len(tl):
        if tl[j] == '__align__' and j + 1 < len(tl) and tl[j + 1] == '(':
            e = _match(tl, j + 1)
            out.extend(['__attribute__', '(', '(', 'aligned', '('] + tl[j + 2:e] + [')', ')', ')'])
            j = e + 1
            continue
        if tl[j] in ('__launch_bounds__', 'alignas', '__declspec') and j + 1 < len(tl) and tl[j + 1] == '(':
            e = _match(tl, j + 1)
            if tl[j] == 'alignas':
                out.extend(['__attribute__', '(', '(', 'aligned', '('] + tl[j + 2:e] + [')', ')', ')'])
            j = e + 1
            continue
        if tl[j] == '[' and j + 1 < len(tl) and tl[j + 1] == '[':
            j = _match(tl, j, '[', ']') + 1
            continue
        out.append(tl[j])
        j += 1
    return out


def _split_commas(tl):
    parts, cur, d = [], [], 0
    for t in tl:
        if t in ('(', '[', '{'):
            d += 1
        elif t in (')', ']', '}'):
            d -= 1
        if t == ',' and d == 0:
            parts.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur or parts:
        parts.append(cur)
    return parts


# ---------------------------------------------------------------------------
# Expression/statement rewriting
# ---------------------------------------------------------------------------
def rewrite(tl, ctx, pp=None):
    """tl: list of token texts (pp lines appear as strings starting with '#')."""
    out = []
    i, n = 0, len(tl)
    P = lambda j: tl[j] if 0 <= j < n else ''
    while i < n:
        t = tl[i]
        if t.startswith('#'):
            body = t.lstrip('#').strip()
            if body.startswith('pragma'):
                out.append('\n#' + body + '\n')
            i += 1
            continue
        if t in IDX_FN:
            if P(i + 1) == '.' and P(i + 2) in ('x', 'y', 'z'):
                out.append('((uint)%s(%d))' % (IDX_FN[t], 'xyz'.index(P(i + 2))))
                i += 3
                continue
            f = IDX_FN[t]
            out.append('pk_make_uint3((uint)%s(0), (uint)%s(1), (uint)%s(2))' % (f, f, f))
            i += 1
            continue
        if t == '__syncthreads' and P(i + 1) == '(' and P(i + 2) == ')':
            out.append('barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)')
            i += 3
            continue
        if t in UNSUPPORTED_IDS:
            raise Unsupported("'%s' is not supported in OpenCL kernels" % t)
        if t in CXX_CASTS and P(i + 1) == '<':
            j = _match_angle(tl, i + 1)
            if P(j + 1) != '(':
                raise Unsupported('cast syntax')
            e = _match(tl, j + 1)
            typ = rewrite(tl[i + 2:j], ctx)
            inner = rewrite(tl[j + 2:e], ctx)
            out.append('((%s)(%s))' % (' '.join(typ), ' '.join(inner)))
            i = e + 1
            continue
        if t == 'std' and P(i + 1) == '::':
            i += 2
            continue
        if t == '::':
            i += 1
            continue
        if IDENT.match(t) and P(i + 1) == '::':   # namespace qualification (namespaces are flattened)
            i += 2
            continue
        if t == 'unsigned' and P(i + 1) == 'long' and P(i + 2) == 'long':
            out.append('ulong')
            i += 3
            if P(i) == 'int':
                i += 1
            continue
        if t == 'long' and P(i + 1) == 'long':
            out.append('long')
            i += 2
            if P(i) == 'int':
                i += 1
            continue
        if t == '__align__' and P(i + 1) == '(':
            e = _match(tl, i + 1)
            out.append('__attribute__((aligned(%s)))' % ' '.join(tl[i + 2:e]))
            i = e + 1
            continue
        if t in ('__launch_bounds__', 'alignas') and P(i + 1) == '(':
            i = _match(tl, i + 1) + 1
            continue
        if t == '[' and P(i + 1) == '[':
            i = _match(tl, i, '[', ']') + 1
            continue
        if t in FUNC_CAST_TYPES and P(i + 1) == '(' and (i == 0 or P(i - 1) in CAST_PREV) \
                and not (P(i - 1) in ('(', ',') and _looks_like_decl(tl, i)):
            e = _match(tl, i + 1)
            inner = rewrite(tl[i + 2:e], ctx)
            ty = {'size_t': 'ulong', 'bool': 'bool'}.get(t, t)
            out.append('((%s)(%s))' % (ty, ' '.join(inner)))
            i = e + 1
            continue
        if t in RENAMES:
            if RENAMES[t]:
                out.append(RENAMES[t])
            i += 1
            continue
        if t == '.' and P(i + 1) in CG_MEMBERS and P(i + 2) == '(' and P(i + 3) == ')' and out:
            # g.sync(), g.thread_rank(), ... on a cooperative_groups::thread_block
            out.pop()
            out.append(CG_MEMBERS[P(i + 1)])
            i += 4
            continue
        if t in ctx.devfuncs and P(i - 1) not in ('.', '->') and P(i + 1) == '(':
            e = _match(tl, i + 1)
            args = _split_commas(tl[i + 2:e]) if e > i + 2 else []
            refs = ctx.refparams.get(t, {}).get(len(args), set())
            parts = []
            for k, a in enumerate(args):
                ra = ' '.join(rewrite(a, ctx))
                parts.append('&(%s)' % ra if k in refs else ra)
            parts += ['pk_s_' + x for x in ctx.sym_uses.get(t, ())]
            out.append('pkd_%s(%s)' % (t, ', '.join(parts)))
            i = e + 1
            continue
        if t in ('sync', 'barrier_wait') and P(i + 1) == '(' and P(i - 1) not in ('.', '->'):
            e = _match(tl, i + 1)   # cooperative_groups::sync(block)
            out.append('barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)')
            i = e + 1
            continue
        if t == 'this_thread_block' and P(i + 1) == '(' and P(i + 2) == ')':
            out.append('pk_this_thread_block()')
            i += 3
            continue
        if t == 'thread_block' and P(i - 1) not in ('.', '->'):
            out.append('pk_thread_block')
            i += 1
            continue
        if ctx.in_body and t in ctx.symbols and P(i - 1) not in ('.', '->', '::'):
            # __device__/__constant__ variables live in buffers passed as hidden arguments
            out.append('(*pk_s_%s)' % t)
            i += 1
            continue
        if t in ctx.deref and P(i - 1) not in ('.', '->'):
            out.append('(*%s)' % t)
            i += 1
            continue
        if t in OCL_RESERVED:
            out.append('pk_r_' + t)
            i += 1
            continue
        out.append(t)
        i += 1
    return out


def _looks_like_decl(tl, i):
    """'float (*f)(int)' style declarators are left alone."""
    return i + 2 < len(tl) and tl[i + 2] == '*'


# ---------------------------------------------------------------------------
# Item conversion
# ---------------------------------------------------------------------------
def _find_param_paren(tl):
    j = 0
    while j < len(tl):
        j2 = _skip_attrs(tl, j)
        if j2 != j:
            j = j2
            continue
        t = tl[j]
        if t == '(':
            return j
        if t in ('=', '{', ';'):
            return -1
        j += 1
    return -1


def classify(item, ctx):
    tl = item.texts()
    if not tl:
        return
    if tl[0] == 'template':
        item.kind = 'template'
        return
    if tl[0] in ('using', 'static_assert'):
        if tl[0] == 'using' and len(tl) > 3 and tl[2] == '=' and tl[1] != 'namespace':
            item.kind = 'using'
            item.name = tl[1]
            item.defines = {tl[1]}
            ctx.typedefs[tl[1]] = [x for x in tl[3:] if x != ';']
        return
    if tl[0] == 'typedef':
        item.kind = 'typedef'
        body = [x for x in tl if x != ';']
        if '{' in body:
            name = body[-1]
            item.defines = {name}
            item.name = name
            if body[1] == 'enum':
                ctx.enums.add(name)
                item.defines |= _enum_constants(body)
            else:
                ctx.structs.add(name)
        else:
            name = body[-1] if IDENT.match(body[-1]) else None
            if name:
                item.defines = {name}
                item.name = name
                ctx.typedefs[name] = body[1:-1]
        return
    head0 = tl[:tl.index('{')] if '{' in tl else tl
    is_type_def = bool(head0) and head0[0] in ('struct', 'class', 'union', 'enum') and '{' in tl and '(' not in head0
    p = -1 if is_type_def else _find_param_paren(tl)
    is_func = p > 0 and IDENT.match(tl[p - 1]) is not None and tl[p - 1] not in TYPE_WORDS
    if is_func:
        # the parameter list must be followed by a body or ';' (otherwise it's e.g. a variable with ctor args)
        e = _match(tl, p)
        rest = tl[e + 1:]
        rest = [x for x in rest if x not in ('const', 'noexcept')] if rest else rest
        if not rest or rest[0] in ('{', ';') or rest[0] in ATTR_FUNCS or rest[0] == '->':
            name = tl[p - 1]
            item.name = name
            if 'operator' in tl[:p]:
                item.kind = 'hostfunc'
                return
            if '__global__' in tl[:p]:
                item.kind = 'kernel' if '{' in tl else 'kernel_decl'
                item.defines = {name}
            elif '__device__' in tl[:p]:
                item.kind = 'devfunc' if '{' in tl else 'devfunc_decl'
                item.defines = {name}
                ctx.devfuncs.add(name)
            else:
                item.kind = 'hostfunc'
            return
    if not is_type_def and any(x in tl for x in ('__device__', '__constant__', '__managed__')):
        item.kind = 'symbol'
        name = _var_name(tl)
        item.name = name
        item.defines = {name} if name else set()
        return
    if '__shared__' in tl:
        item.kind = 'shared_global'
        return
    head = tl[:tl.index('{')] if '{' in tl else tl
    if head and head[0] in ('struct', 'class', 'union', 'enum') and '{' in tl:
        item.kind = 'type'
        name = None
        k = 1
        if k < len(head) and head[k] in ('class', 'struct'):
            k += 1
        k = _skip_attrs(head, k)
        if k < len(head) and IDENT.match(head[k]):
            name = head[k]
        item.name = name
        item.defines = {name} if name else set()
        if head[0] == 'enum':
            ctx.enums.add(name)
            item.defines |= _enum_constants(tl)
        else:
            ctx.structs.add(name)
        return
    if head and head[0] in ('struct', 'class', 'union', 'enum') and len(tl) <= 3:
        item.kind = 'fwd'
        return
    if 'const' in tl[:tl.index('=')] if '=' in tl else False:
        item.kind = 'const'
        name = _var_name(tl)
        item.name = name
        item.defines = {name} if name else set()
        return
    if tl[:1] == ['constexpr'] or ('constexpr' in tl and '=' in tl):
        item.kind = 'const'
        name = _var_name(tl)
        item.name = name
        item.defines = {name} if name else set()
        return


def _enum_constants(tl):
    if '{' not in tl:
        return set()
    b = tl.index('{')
    e = _match(tl, b, '{', '}')
    names = set()
    for part in _split_commas(tl[b + 1:e]):
        if part and IDENT.match(part[0]):
            names.add(part[0])
    return names


def _var_name(tl):
    """Name of a variable declaration: identifier before '[' or '=' or ';' at depth 0."""
    d = 0
    last = None
    for t in tl:
        if t in ('(', '{'):
            if d == 0 and t == '{':
                break
            d += 1
        elif t in (')', '}'):
            d -= 1
        elif d == 0:
            if t in ('[', '=', ';'):
                break
            if IDENT.match(t):
                last = t
    return last


def _param(ptoks, idx, ctx):
    toks = list(ptoks)
    d = 0
    for k, t in enumerate(toks):
        if t in ('(', '[', '{'):
            d += 1
        elif t in (')', ']', '}'):
            d -= 1
        elif t == '=' and d == 0:
            toks = toks[:k]
            break
    toks = _strip_attrs(toks)
    if not toks or toks == ['void']:
        return None
    if '&&' in toks:
        raise Unsupported('rvalue-reference parameter')
    is_ref = False
    if '&' in toks:
        is_ref = 'const' not in toks
        toks = [t for t in toks if t != '&']
    is_array = False
    if '[' in toks:
        k = toks.index('[')
        toks = toks[:k]
        is_array = True
    name = None
    if len(toks) > 1 and IDENT.match(toks[-1]) and toks[-1] not in TYPE_WORDS and toks[-2] not in ('struct', 'enum', 'union'):
        name = toks[-1]
        toks = toks[:-1]
    if name is None:
        name = 'pk_unnamed_%d' % idx
    if name in OCL_RESERVED:
        name = 'pk_r_' + name
    if is_array:
        toks = toks + ['*']
    if '(' in toks:
        raise Unsupported('function-pointer parameter')
    if is_ref:
        if '*' in toks:
            raise Unsupported('reference to pointer parameter')
        return toks + ['*'], name, True
    return toks, name, False


def _scalar(type_words, ctx):
    words = [w for w in type_words if w not in ('const', 'volatile', '__restrict__', '__restrict', 'restrict')]
    for _ in range(6):
        if len(words) == 1 and words[0] in ctx.typedefs:
            words = [w for w in ctx.typedefs[words[0]] if w not in ('const', 'volatile')]
        else:
            break
    if len(words) == 1 and words[0] in ctx.enums:
        return ('int', 'I4')
    c = _canon(words)
    if c in _INT:
        return _INT[c]
    if c in _FLT:
        return _FLT[c]
    return None



_VEC_TYPE = re.compile(r'^(pk_)?(u?(char|short|int|long)|float|double|u?longlong|uchar|ushort|uint|ulong)[1-4]$')
_DECL_OK = {'*', 'const', 'volatile', 'restrict', '__restrict__', '__restrict', 'unsigned', 'signed', 'struct',
            'union', 'enum'}
_AS_WORDS = ('__global', '__local', '__private', '__constant', 'global', 'local', 'private', 'constant')


def _typeish(t, ctx):
    return (t in TYPE_WORDS or t in ctx.typedefs or t in ctx.structs or t in ctx.enums or t in RENAMES
            or _VEC_TYPE.match(t) is not None or t in _INT_TYPE_NAMES)


def _infer_as(tl, gptr, lptr, ctx):
    """Label pointer declarations and pointer casts with __global/__local when the value
    comes from a known global or local pointer. Required by OpenCL C 1.2 (no generic
    address space) and harmless for 2.0/3.0. tl is a list of token texts."""
    gptr, lptr = set(gptr), set(lptr)
    n = len(tl)
    ins = {}

    def as_of(seg):
        ids = set(t for t in seg if IDENT.match(t))
        g, l = bool(ids & gptr), bool(ids & lptr)
        if g and not l:
            return '__global'
        if l and not g:
            return '__local'
        return None

    def type_expr(seg):
        return (seg and _typeish(seg[0], ctx) and '*' in seg and not any(t in _AS_WORDS for t in seg)
                and all(IDENT.match(t) or t in _DECL_OK for t in seg))

    prev = ';'
    i = 0
    while i < n:
        t = tl[i]
        start = prev in (';', '{', '}') or prev.startswith('#') or (prev == '(' and i >= 2 and tl[i - 2] == 'for')
        if start and _typeish(t, ctx):
            # declaration with initializer: TYPE [*...] NAME = EXPR
            k, d = i, 0
            while k < n:
                tk = tl[k]
                if tk in ('(', '['):
                    d += 1
                elif tk in (')', ']'):
                    d -= 1
                elif d == 0 and tk in ('=', ';', ',', '{', '}'):
                    break
                k += 1
            head = tl[i:k]
            if k < n and tl[k] == '=' and len(head) >= 3 and IDENT.match(head[-1]) and type_expr(head[:-1]):
                e, d = k + 1, 0
                while e < n:
                    tk = tl[e]
                    if tk in ('(', '[', '{'):
                        d += 1
                    elif tk in (')', ']', '}'):
                        if d == 0:
                            break
                        d -= 1
                    elif d == 0 and tk in (';', ','):
                        break
                    e += 1
                asp = as_of(tl[k + 1:e])
                if asp:
                    ins.setdefault(i, asp)
                    (gptr if asp == '__global' else lptr).add(head[-1])
        if t in CXX_CASTS and i + 1 < n and tl[i + 1] == '<':
            try:
                j = _match_angle(tl, i + 1)
                if j + 1 < n and tl[j + 1] == '(' and type_expr(tl[i + 2:j]):
                    asp = as_of(tl[j + 2:_match(tl, j + 1)])
                    if asp:
                        ins.setdefault(i + 2, asp)
            except Unsupported:
                pass
        elif t == '(' and i + 1 < n and _typeish(tl[i + 1], ctx):
            try:
                j = _match(tl, i)
            except Unsupported:
                j = -1
            if j > 0 and type_expr(tl[i + 1:j]) and j + 1 < n:
                op = tl[j + 1]
                seg = None
                if op == '(':
                    seg = tl[j + 2:_match(tl, j + 1)]
                elif op == '&' and j + 2 < n:
                    seg = [tl[j + 2]]
                elif IDENT.match(op):
                    seg = [op]
                if seg is not None:
                    asp = as_of(seg)
                    if asp:
                        ins.setdefault(i + 1, asp)
        if not t.startswith('#') or True:
            prev = t
        i += 1
    if not ins:
        return list(tl)
    out = []
    for idx, tok in enumerate(tl):
        if idx in ins:
            out.append(ins[idx])
        out.append(tok)
    return out


def _sym_param(ctx, nm):
    ty, dims = ctx.symbols[nm]
    return '__global %s (*pk_s_%s)%s' % (ty, nm, dims)


def _sym_kernel_param(ctx, nm):
    ty, dims = ctx.symbols[nm]
    return ('__global uchar* pk_sb_%s' % nm,
            '__global %s (*pk_s_%s)%s = (__global %s (*)%s)pk_sb_%s;' % (ty, nm, dims, ty, dims, nm))


def _body(tl, ctx):
    ctx.in_body = True
    try:
        return rewrite(tl, ctx)
    finally:
        ctx.in_body = False


def convert_function(item, ctx):
    tl_all = item.toks
    tl = [t if k != 'pp' else t for k, t in tl_all]
    p = _find_param_paren(tl)
    e = _match(tl, p)
    head = _strip_attrs([t for t in tl[:p - 1] if not t.startswith('#')])
    if any(t == 'extern' for t in head) and '{' not in tl:
        raise Unsupported('extern declaration')
    is_kernel = '__global__' in head
    head = [t for t in head if t not in DROP_SPEC and not (t.startswith('"'))]
    while head and head[-1] == '::':
        head = head[:-2] if len(head) > 1 else []
    if any(t in ('&', '&&', 'auto') for t in head):
        raise Unsupported('reference or auto return type')
    ret = rewrite(head, ctx)
    after = tl[e + 1:]
    j = 0
    while j < len(after) and after[j] not in ('{', ';'):
        if after[j] == 'const':
            raise Unsupported('member function')
        if after[j] == '->':
            raise Unsupported('trailing return type')
        j += 1
    body = after[j + 1:-1] if j < len(after) and after[j] == '{' else None
    params = []
    for idx, part in enumerate(_split_commas(tl[p + 1:e])):
        r = _param(part, idx, ctx)
        if r:
            params.append(r)
    name = item.name
    if is_kernel:
        if any(r for _, _, r in params):
            raise Unsupported('reference kernel parameter')
        return _kernel(item, ctx, name, [(t, n) for t, n, _ in params], body)
    if body is not None and '__shared__' in body:
        raise Unsupported('__shared__ variable inside a __device__ function')
    ptr_names = [pname for typ, pname, _ in params if '*' in typ]
    syms = list(ctx.sym_uses.get(name, ()))

    def build(space):
        # space: None = generic pointers (OpenCL C 2.0/3.0); else '__global' / '__local' / '' (private)
        out_params = []
        for typ, pname, is_ref in params:
            rt = rewrite(typ, ctx)
            if '*' in typ and space:
                rt = [space] + rt
            out_params.append(' '.join(rt + [pname]))
        out_params += [_sym_param(ctx, x) for x in syms]
        rtype = ' '.join(([space] if space and '*' in ret else []) + ret) or 'void'
        sigtxt = '__attribute__((overloadable)) %s pkd_%s(%s)' % (rtype, name, ', '.join(out_params) or 'void')
        if body is None:
            return sigtxt + ';\n'
        ctx.deref = set(n for _, n, r in params if r)
        try:
            b = body
            if space == '__global':
                b = _infer_as(body, ptr_names + syms, [], ctx)
            elif space == '__local':
                b = _infer_as(body, syms, ptr_names, ctx)
            elif space == '':
                b = _infer_as(body, syms, [], ctx)
            return sigtxt + ' {\n' + _fmt(_body(b, ctx)) + '\n}\n'
        finally:
            ctx.deref = set()

    if not ptr_names and '*' not in ret:
        # one copy; with device variables, label pointers taken from them __global (valid in every dialect)
        return build('' if syms else None)
    # Without the generic address space (OpenCL C 1.2, e.g. NVIDIA) emit one copy per address
    # space; overload resolution picks the right one at each call.
    return ('\n#if defined(PK_GENERIC)\n' + build(None) + '#else\n' + build('__global') + build('__local') +
            build('') + '#endif\n')


def _kernel(item, ctx, name, params, body):
    if body is None:
        raise Unsupported('kernel declaration without body')
    cl_params, prologue, sig = [], [], []
    for typ, pname in params:
        if '*' in typ:
            if typ.count('*') > 1:
                raise Unsupported('pointer-to-pointer kernel parameter')
            rt = rewrite(typ, ctx)
            cast_t = [x for x in rt if x != 'restrict']
            cl_params.append('__global %s %s__pkb' % (' '.join(rt), pname))
            cl_params.append('ulong %s__pko' % pname)
            prologue.append('__global %s %s = (__global %s)((__global uchar*)%s__pkb + %s__pko);' % (
                ' '.join(rt), pname, ' '.join(cast_t), pname, pname))
            sig.append('P')
        else:
            sc = _scalar(typ, ctx)
            if sc:
                cl_params.append('%s%s %s' % ('const ' if 'const' in typ else '', sc[0], pname))
                sig.append(sc[1])
            else:
                cl_params.append('%s %s' % (' '.join(rewrite(typ, ctx)), pname))
                sig.append('S')
    syms = list(ctx.sym_uses.get(name, ()))
    for x in syms:   # __device__/__constant__ variables: one buffer argument each (after the user's)
        prm, pro = _sym_kernel_param(ctx, x)
        cl_params.append(prm)
        prologue.append(pro)
    item.syms = syms
    cl_params.append('__local long* pk_dynsmem__')
    # hoist __shared__ declarations to kernel scope (OpenCL requires __local at function scope)
    hoisted, rest, shared_names = _hoist_shared(body, ctx)
    gnames = [pname for typ, pname in params if '*' in typ] + syms
    rest = _infer_as(rest, gnames, list(shared_names) + ['pk_dynsmem__'], ctx)
    text = '__kernel void %s(%s) {\n' % (name, ', '.join(cl_params))
    if prologue:
        text += '  ' + '\n  '.join(prologue) + '\n'
    if hoisted:
        text += '  ' + '\n  '.join(hoisted) + '\n'
    text += _fmt(_body(rest, ctx)) + '\n}\n'
    item.sig = ','.join(sig)
    return text


def _hoist_shared(body, ctx):
    ranges = []
    last = -1
    i = 0
    while i < len(body):
        t = body[i]
        if t.startswith('#') or t in (';', '{', '}'):
            last = i
        elif t == '__shared__':
            e = i
            while e < len(body) and body[e] != ';':
                e += 1
            ranges.append((last + 1, e))
            last = e
            i = e + 1
            continue
        i += 1
    hoisted = []
    rest = []
    names = set()
    pos = 0
    for a, b in ranges:
        rest.extend(body[pos:a])
        stmt = [t for t in body[a:b] if t not in ('static', '__shared__')]
        if 'extern' in stmt:
            stmt = _strip_attrs([t for t in stmt if t != 'extern'])
            if '[' not in stmt:
                raise Unsupported('cannot parse extern __shared__')
            k = stmt.index('[')
            nm = stmt[k - 1]
            ty = ' '.join(rewrite(stmt[:k - 1], ctx))
            hoisted.append('__local %s* %s = (__local %s*)pk_dynsmem__;' % (ty, nm, ty))
            names.add(nm)
        else:
            hoisted.append('__local ' + ' '.join(rewrite(stmt, ctx)) + ';')
            for decl in _split_commas(_strip_attrs(stmt)):
                cut = decl.index('[') if '[' in decl else (decl.index('=') if '=' in decl else len(decl))
                if cut > 0 and IDENT.match(decl[cut - 1]):
                    names.add(decl[cut - 1])
        pos = b + 1
    rest.extend(body[pos:])
    return hoisted, rest, names


def _fmt(out):
    s = ''
    for t in out:
        if t.startswith('\n#'):
            s += t
            continue
        s += t
        if t in (';', '{', '}'):
            s += '\n'
        else:
            s += ' '
    return s


def convert_type(item, ctx):
    tl = [t for t in item.texts()]
    kw = tl[0]
    b = tl.index('{')
    e = _match(tl, b, '{', '}')
    head = _strip_attrs(tl[:b])
    if kw == 'enum':
        if len(head) > 1 and head[1] in ('class', 'struct'):
            raise Unsupported('scoped enum')
        if ':' in head:
            head = head[:head.index(':')]
        name = item.name
        body = ' '.join(rewrite(tl[b + 1:e], ctx))
        if name:
            return 'typedef enum %s { %s } %s;\n' % (name, body, name)
        return 'enum { %s };\n' % body
    if ':' in head:
        raise Unsupported('struct with base class')
    attrs = [t for t in tl[1:b] if t not in (item.name,)]
    attr_txt = ''
    if 'alignas' in tl[:b] or '__align__' in tl[:b] or '__attribute__' in tl[:b]:
        attr_txt = ' '.join(_strip_attrs([t for t in tl[1:b] if t != item.name]))
    members = _struct_members(tl[b + 1:e], ctx)
    name = item.name or 'pk_anon_%d' % id(item)
    kwd = 'union' if kw == 'union' else 'struct'
    return 'typedef %s %s %s { %s } %s;\n' % (kwd, attr_txt, name, ' '.join(members), name)


def _struct_members(tl, ctx):
    out = []
    cur = []
    i = 0
    while i < len(tl):
        t = tl[i]
        if t in ('public', 'private', 'protected') and i + 1 < len(tl) and tl[i + 1] == ':':
            i += 2
            continue
        if t == '{':   # member function body or brace initializer
            e = _match(tl, i, '{', '}')
            if '(' in cur and '=' not in cur:   # method / ctor definition
                cur = []
                i = e + 1
                if i < len(tl) and tl[i] == ';':
                    i += 1
                continue
            cur.extend(tl[i:e + 1])
            i = e + 1
            continue
        if t == ';':
            m = _member(cur, ctx)
            if m:
                out.append(m)
            cur = []
            i += 1
            continue
        cur.append(t)
        i += 1
    return out


def _member(cur, ctx):
    if not cur:
        return None
    if cur[0] in ('static', 'using', 'typedef', 'friend', 'template', 'enum', 'struct', 'union', 'class'):
        if cur[0] in ('struct', 'union', 'enum', 'class') and '{' in cur:
            raise Unsupported('nested type in struct')
        if cur[0] in ('struct', 'union') and '(' not in cur:
            pass
        else:
            return None
    if '(' in cur and '=' not in cur:   # method declaration / ctor
        p = cur.index('(')
        if p > 0 and (IDENT.match(cur[p - 1]) or cur[p - 1] == 'operator'):
            return None
    d = 0
    cut = len(cur)
    for k, t in enumerate(cur):
        if t in ('(', '[', '{'):
            if t == '{' and d == 0:
                cut = k
                break
            d += 1
        elif t in (')', ']', '}'):
            d -= 1
        elif t == '=' and d == 0:
            cut = k
            break
    cur = cur[:cut]
    if '&' in cur:
        raise Unsupported('reference member')
    return ' '.join(rewrite([t for t in cur if t not in DROP_SPEC], ctx)) + ';'


def convert_typedef(item, ctx):
    tl = item.texts()
    if '{' in tl:
        b = tl.index('{')
        kw = tl[1]
        e = _match(tl, b, '{', '}')
        if kw == 'enum':
            return 'typedef enum { %s } %s;\n' % (' '.join(rewrite(tl[b + 1:e], ctx)), item.name)
        members = _struct_members(tl[b + 1:e], ctx)
        tag = tl[2] if b > 2 and IDENT.match(tl[2]) else ''
        return 'typedef %s %s { %s } %s;\n' % (kw if kw in ('struct', 'union') else 'struct', tag,
                                              ' '.join(members), item.name)
    if '<' in tl or '(' in tl:
        raise Unsupported('complex typedef')
    return ' '.join(rewrite(tl, ctx)) + (';' if tl[-1] != ';' else '') + '\n'


def convert_using(item, ctx):
    tl = item.texts()
    ty = [x for x in tl[3:] if x != ';']
    if '<' in ty or '(' in ty:
        raise Unsupported('complex using alias')
    return 'typedef %s %s;\n' % (' '.join(rewrite(ty, ctx)), tl[1])


def convert_const(item, ctx):
    tl = [t for t in item.texts() if t != ';']
    tl = [t for t in tl if t not in ('static', 'constexpr', 'inline', 'extern', '__host__', '__device__')]
    if '=' not in tl:
        raise Unsupported('const without initializer')
    eq = tl.index('=')
    decl = tl[:eq]
    init = tl[eq + 1:]
    name = item.name
    if '[' in decl:
        ty = decl[:decl.index(name)]
        dims = decl[decl.index(name) + 1:]
        return '__constant %s %s%s = %s;\n' % (' '.join(rewrite(ty, ctx)), name, ' '.join(rewrite(dims, ctx)),
                                              ' '.join(rewrite(init, ctx)))
    ty = [t for t in decl[:decl.index(name)] if t != 'const']
    sc = _scalar(ty, ctx)
    if sc and sc[1].startswith('I') and ty != ['bool']:
        return 'enum { %s = (%s) };\n' % (name, ' '.join(rewrite(init, ctx)))
    if '*' in ty:
        raise Unsupported('pointer constant')
    return '__constant %s %s = %s;\n' % (' '.join(rewrite(ty, ctx)), name, ' '.join(rewrite(init, ctx)))


def convert_symbol(item, ctx):
    tl = [t for t in item.texts() if t != ';']
    if 'extern' in tl:
        raise Unsupported('extern device variable (separate compilation)')
    if '__managed__' in tl:
        raise Unsupported('__managed__ variable')
    tl = _strip_attrs([t for t in tl if t not in ('__device__', '__constant__', 'static', 'volatile')])
    d = 0
    for t in tl:
        if t in ('(', '[', '{'):
            d += 1
        elif t in (')', ']', '}'):
            d -= 1
        elif t == ',' and d == 0:
            raise Unsupported('multiple declarators')
    if '&' in tl:
        raise Unsupported('reference variable')
    if '=' in tl:   # the initial value comes from the host copy of the variable
        tl = tl[:tl.index('=')]
    k = tl.index('[') if '[' in tl else len(tl)
    if '(' in tl[:k]:
        raise Unsupported('constructor-initialised device variable')
    if k < 2 or not IDENT.match(tl[k - 1]):
        raise Unsupported('cannot parse device variable')
    if '*' in tl[:k]:
        raise Unsupported('pointer device variable')
    name = tl[k - 1]
    ty = ' '.join(rewrite(tl[:k - 1], ctx))
    dims = ' '.join(rewrite(tl[k:], ctx))
    ctx.symbols[name] = (ty, dims)
    return '/* __device__ variable %s: hidden kernel argument pk_s_%s */\n' % (name, name)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def find_clang():
    if os.environ.get('PK_CLANG') == 'none':   # skip validation (the device driver compiles unchecked code)
        return None
    for c in (os.environ.get('PK_CLANG'), 'clang', 'clang-20', 'clang-19', 'clang-18', 'clang-17', 'clang-16',
              'clang-15'):
        if c and shutil.which(c):
            return shutil.which(c)
    return None


def validate(src, clang, std='CL3.0'):
    with tempfile.NamedTemporaryFile('w', suffix='.cl', delete=False, encoding='utf-8') as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run([clang, '-x', 'cl', '-cl-std=' + std, '-Xclang', '-finclude-default-header',
                            '-fsyntax-only', '-target', 'spir64', '-Wno-everything', '-ferror-limit=50', path],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', errors='replace')
    finally:
        os.unlink(path)
    errs = []
    for line in r.stderr.splitlines():
        m = re.match(r'^.*?\.cl:(\d+):(\d+): error: (.*)$', line)
        if m:
            errs.append((int(m.group(1)), m.group(3)))
    if r.returncode != 0 and not errs:
        errs.append((0, r.stderr.strip()[:300]))
    return errs


def generate(dev_src, log=None):
    toks = [t for t in tokenize(dev_src) if t[0] not in ('ws', 'nl')]
    needs_fibers = any(t[0] == 'id' and t[1] in SYNC_IDS for t in toks)
    result = {'needs_fibers': needs_fibers, 'cl_ok': False, 'cl_src': None, 'kernels': [], 'symbols': [],
              'reason': None, 'skipped': []}
    items = split_items(toks)
    ctx = Ctx()
    for it in items:
        try:
            classify(it, ctx)
        except Unsupported as e:
            it.kind = 'other'
    for it in items:
        if it.kind in ('devfunc', 'devfunc_decl'):
            try:
                tl = it.texts()
                p = _find_param_paren(tl)
                e = _match(tl, p)
                prm = [_param(part, k, ctx) for k, part in enumerate(_split_commas(tl[p + 1:e]))]
                prm = [x for x in prm if x]
                ctx.refparams.setdefault(it.name, {})[len(prm)] = set(k for k, x in enumerate(prm) if x[2])
            except Unsupported:
                pass
    for it in items:
        it.refs = set(t for k, t in it.toks if k == 'id')
    # __device__/__constant__ variables become buffers passed to every kernel and device function
    # that uses them (directly or through calls); works on OpenCL C 1.2 as well as 2.0/3.0.
    for it in items:
        if it.kind == 'symbol':
            try:
                it.cl = convert_symbol(it, ctx)
                if it.name:
                    ref = '::' + '::'.join([x for x in it.ns if x != '(anon)'] + [it.name])
                    it.symbol = (ref, it.name)
            except Unsupported as e:
                it.error = str(e)
                it.cl = None
    funcs = {}
    for it in items:
        if it.kind in ('kernel', 'devfunc', 'devfunc_decl') and it.name:
            funcs.setdefault(it.name, set()).update(it.refs)
    uses = {f: refs & set(ctx.symbols) for f, refs in funcs.items()}
    changed = True
    while changed:
        changed = False
        for f, refs in funcs.items():
            for g in refs & ctx.devfuncs:
                if g in uses and not uses[g] <= uses[f]:
                    uses[f] |= uses[g]
                    changed = True
    ctx.sym_uses = {f: sorted(u) for f, u in uses.items() if u}
    for it in items:
        try:
            if it.kind == 'symbol':
                continue
            if it.kind in ('kernel', 'devfunc', 'devfunc_decl'):
                it.cl = convert_function(it, ctx)
            elif it.kind == 'type':
                it.cl = convert_type(it, ctx)
            elif it.kind == 'typedef':
                it.cl = convert_typedef(it, ctx)
            elif it.kind == 'using':
                it.cl = convert_using(it, ctx)
            elif it.kind == 'const':
                it.cl = convert_const(it, ctx)
        except Unsupported as e:
            it.error = str(e)
            it.cl = None
    kernels = [it for it in items if it.kind == 'kernel']
    if not kernels:
        result['cl_ok'] = True
        result['reason'] = 'no kernels'
        result['cl_src'] = None
        return result

    removed_names = set()
    clang = find_clang()
    with open(os.path.join(HERE, 'ocl_prelude.cl'), encoding='utf-8') as f:
        prelude = f.read()
    for attempt in range(40):
        # dependency closure from kernels and symbols
        defmap = {}
        for it in items:
            for d in it.defines:
                defmap.setdefault(d, []).append(it)
        for it in items:
            it.included = False
        work = [it for it in items if it.kind in ('kernel', 'symbol') and it.cl and not it.error]
        for it in work:
            it.included = True
        while work:
            it = work.pop()
            for r in it.refs:
                for dep in defmap.get(r, []):
                    if not dep.included and dep is not it:
                        dep.included = True
                        work.append(dep)
        # drop items that depend on failed items
        changed = True
        while changed:
            changed = False
            for it in items:
                if it.included and (it.error or it.cl is None) and it.kind not in ('hostfunc', 'other', 'template',
                                                                                  'fwd'):
                    pass
            bad = set()
            for it in items:
                if it.error and it.defines:
                    bad |= it.defines
            for it in items:
                if it.included and not it.error and (it.refs & bad) and not (it.defines & bad):
                    # only an error if the bad name is not also defined by a good item (overloads)
                    good_defs = set()
                    for r in it.refs & bad:
                        if any(not d.error for d in defmap.get(r, []) if d.cl):
                            good_defs.add(r)
                    if (it.refs & bad) - good_defs:
                        it.error = 'depends on unsupported %s' % ', '.join(sorted((it.refs & bad) - good_defs))
                        changed = True
        parts = [prelude, '\n']
        line_map = []
        line = prelude.count('\n') + 2
        for it in items:
            if not it.included or it.error or not it.cl:
                continue
            text = it.cl if it.cl.endswith('\n') else it.cl + '\n'
            line_map.append((line, line + text.count('\n'), it))
            parts.append(text)
            line += text.count('\n')
        src = ''.join(parts)
        if not clang:
            break
        errs = validate(src, clang)
        if not errs:
            break
        ln, msg = errs[0]
        culprit = None
        for a, b, it in line_map:
            if a <= ln < b:
                culprit = it
                break
        if culprit is None:
            result['reason'] = 'OpenCL validation failed: %s' % msg
            return result
        culprit.error = 'OpenCL C: ' + msg
        if log:
            log('device pass: dropped %s %s (%s)' % (culprit.kind, culprit.name, msg))
    else:
        result['reason'] = 'too many OpenCL translation errors'
        return result

    ok_kernels = [it for it in items if it.kind == 'kernel' and it.included and not it.error and it.cl]
    result['kernels'] = [(it.name, it.sig, getattr(it, 'syms', [])) for it in ok_kernels]
    # every translatable variable, used by a kernel or not: the host may still copy to/from it
    result['symbols'] = [it.symbol for it in items if it.kind == 'symbol' and not it.error and it.symbol]
    result['skipped'] = [(it.kind, it.name, it.error) for it in items
                         if it.error and it.kind in ('kernel', 'devfunc', 'symbol', 'type')]
    result['cl_src'] = src
    result['cl_ok'] = True
    if clang:   # informational: will it build on OpenCL C 1.2 devices (no generic address space)?
        e12 = validate(src, clang, 'CL1.2')
        result['cl12'] = 'ok' if not e12 else ('line %d: %s' % e12[0])
    failed = [it for it in kernels if it.error]
    if failed:
        result['reason'] = '; '.join('%s: %s' % (it.name, it.error) for it in failed[:3])
    return result
