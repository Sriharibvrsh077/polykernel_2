"""PolyKernel's own CUDA -> PTX compiler.

Compiles the device code of one translation unit (the device-pass preprocessed source)
into a PTX module for NVIDIA GPUs, written from NVIDIA's public PTX ISA documentation.
The runtime (runtime/pk_ptx.cpp) loads the module through the NVIDIA driver API.

Design:
  * pk_cparse parses each top-level item into an AST with C types.
  * Device functions are inlined into kernels (no PTX calling convention needed); pointer
    provenance (global/shared/local/const) is tracked per inlined instance so loads and
    stores use the right state space, and pointer values are always generic addresses.
  * Scalars live in PTX virtual registers; small structs and arrays indexed only by
    constants are split into registers; everything else lives in a per-kernel local depot.
  * #pragma unroll loops with constant bounds are unrolled with the loop variable as a
    compile-time constant.
  * Math functions that have no PTX instruction come from ptx_prelude.cu, compiled by this
    same compiler.
"""
import os
import re
import shutil
import struct
import subprocess

from pk_lex import tokenize
from pk_cparse import (Parser, ParseError, CType, TVoid, TInt, TFloat, TPtr, TRef, TArr, TStruct, TFunc, TGroup,
                       TFunctor, VOID, BOOL, CHAR, SCHAR, UCHAR, SHORT, USHORT, INT, UINT, LONG, ULONG, LLONG, ULLONG,
                       FLOAT, DOUBLE, TYPEDEFS, VEC, VarDecl, FuncDef, ENum, EStr, EName, EMember, EIndex, ECall,
                       EUnary, EBin, EAssign, ECond, ECast, ESizeof, EComma, EInit, EType, EAsm, SBlock, SExpr, SDecl,
                       SIf, SWhile, SDo, SFor, SSwitch, SCase, SDefault, SBreak, SContinue, SReturn, SEmpty, f32_bits,
                       f64_bits, round_f32, unqual)

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_LEAVES = 64          # aggregates with more scalar leaves stay in memory
MAX_UNROLL = 1024


class CompileError(Exception):
    pass


CG_NS = ('cooperative_groups', 'cg')
STRIP_NS = ('std', 'cooperative_groups', 'cg', 'cuda', 'cuda::std', 'detail')
GROUP_TYPES = {'thread_block': 'block', 'thread_group': 'group', 'grid_group': 'grid', 'coalesced_group': 'coalesced'}


# ---------------------------------------------------------------------------
# Environment: global names known while parsing and compiling
# ---------------------------------------------------------------------------
class Env:
    def __init__(self):
        self.types = {}
        self.structs = {}
        self.enum_consts = {}    # name -> (value, type)
        self.consts = {}         # name -> VarDecl (file-scope const / constexpr)
        self.globals = {}        # name -> VarDecl (__device__/__constant__/__shared__ variables)
        self.funcs = {}          # name -> [FuncDef]
        self.locals = []

    def _candidates(self, name, ns):
        out = []
        ns = list(ns or [])
        for k in range(len(ns), -1, -1):
            out.append('::'.join(ns[:k] + [name]))
        parts = name.split('::')
        while len(parts) > 1 and parts[0] in STRIP_NS + ('',):
            parts = parts[1:]
            out.append('::'.join(parts))
        if len(parts) > 1 and '::'.join(parts[:-1]) in STRIP_NS:
            out.append(parts[-1])
        return out

    def lookup_type(self, name, ns=()):
        if not name:
            return None
        for c in self._candidates(name, ns):
            if c in self.types:
                return self.types[c]
            if c in self.structs:
                return self.structs[c]
            if c in TYPEDEFS:
                return TYPEDEFS[c]
            if c in VEC:
                return VEC[c]
            if c in GROUP_TYPES:
                return TGroup(GROUP_TYPES[c])
        return None

    def lookup_struct(self, name, ns=(), local=False):
        for c in self._candidates(name, ns):
            if c in self.structs:
                return self.structs[c]
        return None

    def define_struct(self, name, t, ns=()):
        self.structs['::'.join(list(ns) + [name])] = t
        if ns:
            self.structs.setdefault(name, t)

    def define_typedef(self, name, t, ns=()):
        self.types['::'.join(list(ns) + [name])] = t
        if ns:
            self.types.setdefault(name, t)

    def define_enum_const(self, name, val, t, ns=(), scope=None):
        full = '::'.join(list(ns) + ([scope] if scope else []) + [name])
        self.enum_consts[full] = (val, t)
        if ns or scope:
            self.enum_consts.setdefault(('%s::%s' % (scope, name)) if scope else name, (val, t))

    def push_scope(self):
        self.locals.append(set())

    def pop_scope(self):
        self.locals.pop()

    def declare_local(self, name):
        if self.locals:
            self.locals[-1].add(name)

    def is_local(self, name):
        return any(name in s for s in self.locals)

    def lookup_value(self, name, ns=()):
        for c in self._candidates(name, ns):
            if c in self.enum_consts:
                return 'enum', self.enum_consts[c]
            if c in self.globals:
                return 'global', self.globals[c]
            if c in self.consts:
                return 'const', self.consts[c]
        return None, None

    def lookup_funcs(self, name, ns=()):
        for c in self._candidates(name, ns):
            if c in self.funcs:
                return self.funcs[c]
        return None

    # constant expressions (array sizes, enum values, template arguments)
    def const_eval(self, e, cenv=None):
        v = const_value(e, self, cenv or {})
        if v is None:
            raise ParseError('not a constant expression')
        return v[0]


def _wrap(v, t):
    if t.is_float():
        return round_f32(v) if t.size == 4 else float(v)
    if t is BOOL:
        return 1 if v else 0
    bits = t.size * 8
    v &= (1 << bits) - 1
    if t.signed and v >= 1 << (bits - 1):
        v -= 1 << bits
    return v


def promote(t):
    if t.is_int() and t.rank < 3:
        return INT
    return t


def common_type(a, b):
    if a.is_float() or b.is_float():
        if (a.is_float() and a.size == 8) or (b.is_float() and b.size == 8):
            return DOUBLE
        return FLOAT
    a, b = promote(a), promote(b)
    if a is b:
        return a
    if a.signed == b.signed:
        return a if a.rank >= b.rank else b
    s, u = (a, b) if a.signed else (b, a)
    if u.rank >= s.rank:
        return u
    if s.size > u.size:
        return s
    return {INT: UINT, LONG: ULONG, LLONG: ULLONG}.get(s, u)


def const_value(e, env, cenv):
    """Fold a constant expression. Returns (value, type) or None. cenv: name -> (value, type)."""
    if isinstance(e, ENum):
        return e.v, e.t
    if isinstance(e, EName):
        if e.name in cenv:
            return cenv[e.name]
        if cenv.get('__locals__') and e.name in cenv['__locals__']:
            return None
        kind, v = env.lookup_value(e.name, cenv.get('__ns__', ()))
        if kind == 'enum':
            return v
        if kind == 'const' and v.init is not None and not isinstance(v.init, EInit) and v.t.is_scalar():
            r = const_value(v.init, env, {})
            if r is not None:
                return _wrap(r[0], v.t) if v.t.is_arith() else r[0], v.t
        if e.name == 'warpSize':
            return 32, INT
        return None
    if isinstance(e, ESizeof):
        if e.t is not None:
            return e.t.size, ULONG
        return None
    if isinstance(e, ECast):
        if isinstance(e.e, (list, EInit)) or not (e.t.is_arith()):
            return None
        r = const_value(e.e, env, cenv)
        if r is None:
            return None
        v = r[0]
        if e.t.is_int() and isinstance(v, float):
            if v != v or abs(v) == float('inf'):
                return None
            v = int(v)
        return _wrap(v, e.t), e.t
    if isinstance(e, EUnary) and e.op in ('-', '+', '~', '!'):
        r = const_value(e.e, env, cenv)
        if r is None:
            return None
        v, t = r
        t = promote(t) if t.is_int() else t
        if e.op == '-':
            return _wrap(-v, t), t
        if e.op == '+':
            return v, t
        if e.op == '~':
            return (_wrap(~v, t), t) if t.is_int() else None
        return (0 if v else 1), BOOL
    if isinstance(e, EBin):
        a = const_value(e.a, env, cenv)
        if a is None:
            return None
        if e.op in ('&&', '||'):
            if e.op == '&&' and not a[0]:
                return 0, BOOL
            if e.op == '||' and a[0]:
                return 1, BOOL
            b = const_value(e.b, env, cenv)
            return None if b is None else ((1 if b[0] else 0), BOOL)
        b = const_value(e.b, env, cenv)
        if b is None:
            return None
        return fold_bin(e.op, a, b)
    if isinstance(e, ECond):
        c = const_value(e.c, env, cenv)
        if c is None:
            return None
        return const_value(e.a if c[0] else e.b, env, cenv)
    return None


def fold_bin(op, a, b):
    (va, ta), (vb, tb) = a, b
    if op in ('<<', '>>'):
        t = promote(ta)
        if not t.is_int() or not tb.is_int() or vb < 0 or vb >= t.size * 8:
            return None
        if op == '<<':
            return _wrap(va << vb, t), t
        return _wrap(va >> vb, t) if t.signed else _wrap((va & ((1 << t.size * 8) - 1)) >> vb, t), t
    t = common_type(ta, tb)
    if t.is_int():
        va, vb = _wrap(int(va), t), _wrap(int(vb), t)
    else:
        va, vb = float(va), float(vb)
    if op in ('==', '!=', '<', '>', '<=', '>='):
        r = {'==': va == vb, '!=': va != vb, '<': va < vb, '>': va > vb, '<=': va <= vb, '>=': va >= vb}[op]
        return (1 if r else 0), BOOL
    if op == '+':
        return _wrap(va + vb, t), t
    if op == '-':
        return _wrap(va - vb, t), t
    if op == '*':
        return _wrap(va * vb, t), t
    if op in ('/', '%'):
        if vb == 0:
            return None
        if t.is_float():
            return (_wrap(va / vb, t), t) if op == '/' else None
        q = abs(va) // abs(vb)
        if (va < 0) != (vb < 0):
            q = -q
        return (_wrap(q, t) if op == '/' else _wrap(va - q * vb, t)), t
    if t.is_float():
        return None
    if op == '&':
        return _wrap(va & vb, t), t
    if op == '|':
        return _wrap(va | vb, t), t
    if op == '^':
        return _wrap(va ^ vb, t), t
    return None


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------
class V:
    """A scalar rvalue: a register or a compile-time constant."""
    __slots__ = ('t', 'r', 'imm', 'space')

    def __init__(self, t, r=None, imm=None, space=None):
        self.t = t
        self.r = r
        self.imm = imm
        self.space = space

    @property
    def is_imm(self):
        return self.r is None

    @property
    def is_pred(self):
        return self.r is not None and self.r.startswith('%p')


class A:
    """An aggregate rvalue held as leaves (path -> V)."""
    __slots__ = ('t', 'leaves')

    def __init__(self, t, leaves):
        self.t = t
        self.leaves = leaves


class GroupV:
    __slots__ = ('t', 'mask', 'size')

    def __init__(self, t, mask=None, size=None):
        self.t = t
        self.mask = mask
        self.size = size


class LReg:
    """A variable (or part of one) held in registers."""
    __slots__ = ('t', 'store', 'path', 'space')

    def __init__(self, t, store, path=(), space=None):
        self.t = t
        self.store = store      # path -> register name
        self.path = path
        self.space = space      # provenance for pointer variables


class LMem:
    """An object in memory: [base + off] in a state space."""
    __slots__ = ('t', 'space', 'base', 'off', 'readonly')

    def __init__(self, t, space, base, off=0, readonly=False):
        self.t = t
        self.space = space      # global shared local const param generic
        self.base = base        # symbol name or 64-bit register holding an address of that space
        self.off = off
        self.readonly = readonly


def leaves_of(t, prefix=(), off=0, out=None):
    if out is None:
        out = []
    if isinstance(t, TStruct):
        if t.union:
            raise CompileError('union')
        for name, ft, fo in t.fields:
            if name is None:
                raise CompileError('anonymous member')
            leaves_of(ft, prefix + (name,), off + fo, out)
    elif isinstance(t, TArr):
        if t.n is None:
            raise CompileError('unsized array')
        for i in range(t.n):
            leaves_of(t.elem, prefix + (i,), off + i * t.elem.size, out)
            if len(out) > MAX_LEAVES * 4:
                raise CompileError('too large')
    else:
        out.append((prefix, t, off))
    return out


def small_agg(t):
    try:
        return len(leaves_of(t)) <= MAX_LEAVES
    except CompileError:
        return False


def sub_type(t, path):
    for p in path:
        if isinstance(p, int):
            t = t.elem
        else:
            t = t.field(p)[1]
    return t


# ---------------------------------------------------------------------------
# PTX module and function emitters
# ---------------------------------------------------------------------------
def mem_type(t):
    if t is BOOL:
        return 'u8'
    if isinstance(t, TPtr):
        return 'u64'
    if t.is_float():
        return 'f32' if t.size == 4 else 'f64'
    if t.is_int():
        return ('s' if t.signed else 'u') + str(t.size * 8)
    raise CompileError('cannot load/store %r' % (t,))


def op_type(t):
    """Instruction type for arithmetic on (promoted) t."""
    if isinstance(t, TPtr):
        return 'u64'
    if t.is_float():
        return 'f32' if t.size == 4 else 'f64'
    if t.size == 8:
        return 's64' if t.signed else 'u64'
    return 's32' if t.signed else 'u32'


def reg_class(t):
    if isinstance(t, TPtr):
        return 'rd'
    if t.is_float():
        return 'f' if t.size == 4 else 'fd'
    if t.size == 8:
        return 'rd'
    return 'r'


def fmt_imm(v, t):
    if t.is_float():
        return '0f%08X' % f32_bits(v) if t.size == 4 else '0d%016X' % f64_bits(v)
    bits = 64 if isinstance(t, TPtr) else max(32, t.size * 8)
    v = int(v)
    if isinstance(t, TPtr) or not t.signed:
        return str(v & ((1 << bits) - 1))
    return str(v)


def ident(s):
    return re.sub(r'[^A-Za-z0-9_]', '_', s)


class Module:
    def __init__(self, env, target):
        self.env = env
        self.target = target
        self.decls = []           # module-scope declarations (text)
        self.declared = set()
        self.strings = {}
        self.entries = []
        self.kernel_names = []
        self.need_sm = 60
        self.need_ptx = (6, 4)    # minimum PTX ISA version a feature in the module needs
        self.dynsmem = False

    def declare(self, key, text):
        if key not in self.declared:
            self.declared.add(key)
            self.decls.append(text)

    def string(self, data):
        if data not in self.strings:
            name = 'pk_str%d' % len(self.strings)
            self.strings[data] = name
            b = list(data) + [0]
            self.declare(name, '.global .align 1 .b8 %s[%d] = {%s};' % (name, len(b), ', '.join(map(str, b))))
        return self.strings[data]

    def text(self):
        sm = max(self.target, self.need_sm)
        ver = (6, 4) if sm <= 75 else ((7, 0) if sm <= 80 else ((7, 1) if sm <= 86 else (7, 8)))
        ver = '%d.%d' % max(ver, self.need_ptx)
        out = ['// Generated by PolyKernel pk_ptx (CUDA -> PTX)', '.version ' + ver, '.target sm_%d' % sm,
               '.address_size 64', '']
        out += self.decls
        out.append('')
        out += self.entries
        return '\n'.join(out) + '\n'


class Fn:
    def __init__(self):
        self.lines = []
        self.n = {'p': 1, 'r': 1, 'rd': 1, 'f': 1, 'fd': 1}
        self.depot = 0
        self.depot_align = 8
        self.nlabel = 0

    def reg(self, cls):
        k = self.n[cls]
        self.n[cls] = k + 1
        return '%%%s%d' % (cls, k)

    def emit(self, s):
        self.lines.append('    ' + s)

    def label(self):
        self.nlabel += 1
        return '$Lpk%d' % self.nlabel

    def place(self, lab):
        self.lines.append(lab + ':')

    def slot(self, size, align):
        align = max(align, 1)
        self.depot = (self.depot + align - 1) // align * align
        off = self.depot
        self.depot += max(size, 1)
        self.depot_align = max(self.depot_align, align)
        return off


# ---------------------------------------------------------------------------
# Per-instance analysis (address-taken variables, constant indices, pointer provenance)
# ---------------------------------------------------------------------------
JOIN_BOTTOM = None


def join(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a if a == b else 'generic'


# built-ins that write their output arguments directly (the variable behind &x can stay in a register);
# math functions implemented in ptx_prelude.cu take real pointers instead
OUT_PARAM_BUILTINS = {'__sincosf', '__match_all_sync'}


class Analysis:
    def __init__(self, gen, params):
        self.gen = gen
        self.mem = set()           # decl ids that need memory
        self.written = set()       # decl ids assigned after initialisation
        self.prov = {}             # decl id -> provenance of a pointer variable
        self.params = params       # name -> (decl, provenance, is_ref)
        self.scopes = [dict((n, d) for n, (d, p, r) in params.items())]
        for n, (d, p, r) in params.items():
            if p is not None:
                self.prov[d.id] = p
        self.changed = False

    def run(self, body, iterations=4):
        for _ in range(iterations):
            self.changed = False
            self.scopes = [self.scopes[0]]
            self.cenv = {'__ns__': self.gen.ns}
            self.stmt(body)
            if not self.changed:
                break

    def find(self, name):
        for s in reversed(self.scopes):
            if name in s:
                return s[name]
        return None

    def set_prov(self, d, p):
        if d is None or not isinstance(unqual(d.t) if d.t is not None else None, TPtr) and d.t is not None:
            return
        old = self.prov.get(d.id)
        new = join(old, p)
        if new != old:
            self.prov[d.id] = new
            self.changed = True

    def mark_mem(self, d):
        if d is not None and d.id not in self.mem:
            self.mem.add(d.id)
            self.changed = True

    # statements
    def stmt(self, s):
        if s is None:
            return
        if isinstance(s, SBlock):
            self.scopes.append({})
            saved = dict(self.cenv)
            for x in s.stmts:
                self.stmt(x)
            self.cenv = saved
            self.scopes.pop()
        elif isinstance(s, SExpr):
            self.expr(s.e)
        elif isinstance(s, SDecl):
            for d in s.decls:
                if d.init is not None:
                    self.init(d, d.init)
                if d.t is not None and isinstance(d.t, TArr) and not small_agg(d.t):
                    self.mark_mem(d)
                if d.t is not None and isinstance(d.t, TStruct) and not small_agg(d.t):
                    self.mark_mem(d)
                self.scopes[-1][d.name] = d
                cv = self.const_local(d)
                if cv is not None:
                    self.cenv[d.name] = cv
                elif d.name in self.cenv:
                    del self.cenv[d.name]
        elif isinstance(s, SIf):
            self.expr(s.c)
            self.stmt(s.t)
            self.stmt(s.e)
        elif isinstance(s, (SWhile, SDo)):
            self.expr(s.c)
            self.stmt(s.body)
        elif isinstance(s, SFor):
            self.scopes.append({})
            saved = dict(self.cenv)
            self.stmt(s.init)
            info = self.gen.unroll_info(s, self.cenv, self.find_decl_written)
            if info:
                var, values = info
                self.cenv[var] = (0, INT)      # any constant: marks the loop variable as compile-time
                if s.c is not None:
                    pass
                self.stmt(s.body)
            else:
                self.expr(s.c)
                self.expr(s.step)
                self.stmt(s.body)
            self.cenv = saved
            self.scopes.pop()
        elif isinstance(s, SSwitch):
            self.expr(s.e)
            self.stmt(s.body)
        elif isinstance(s, SReturn):
            self.expr(s.e)

    def find_decl_written(self, name, body):
        return name_written(name, body)

    def const_local(self, d):
        if d.t is None or not d.t.is_arith() or d.init is None or isinstance(d.init, EInit):
            return None
        if 'const' not in d.storage and 'constexpr' not in d.storage:
            return None
        r = const_value(d.init, self.gen.env, self.cenv)
        if r is None:
            return None
        return _wrap(r[0], d.t), d.t

    def init(self, d, e):
        self.expr(e)
        if isinstance(e, EInit):
            return
        t = d.t
        if t is None or isinstance(unqual(t), TPtr):
            self.set_prov(d, self.prov_of(e))
        if isinstance(t, TRef):
            r = self.root(e)
            if r is not None:
                pass

    # expressions
    def root(self, e):
        """Local declaration an lvalue expression is part of (through . and [] on arrays)."""
        while True:
            if isinstance(e, EName):
                return self.find(e.name)
            if isinstance(e, EMember) and not e.arrow:
                e = e.obj
            elif isinstance(e, EIndex):
                d = self.root_decl_if_array(e.base)
                if d is None:
                    return None
                e = e.base
            else:
                return None

    def root_decl_if_array(self, e):
        t = self.static_type(e)
        if isinstance(t, TArr):
            return True
        return None

    def static_type(self, e):
        """Best-effort declared type of an lvalue expression (None if unknown)."""
        if isinstance(e, EName):
            d = self.find(e.name)
            if d is not None:
                return unqual(d.t) if d.t is not None else None
            kind, g = self.gen.env.lookup_value(e.name, self.gen.ns)
            if kind in ('global', 'const'):
                return g.t
            return None
        if isinstance(e, EMember):
            t = self.static_type(e.obj)
            if t is None:
                return None
            if e.arrow:
                t = t.to if isinstance(t, TPtr) else None
            if isinstance(t, TStruct) and isinstance(e.name, str):
                f = t.field(e.name)
                return f[1] if f else None
            return None
        if isinstance(e, EIndex):
            t = self.static_type(e.base)
            if isinstance(t, TArr):
                return t.elem
            if isinstance(t, TPtr):
                return t.to
            return None
        if isinstance(e, EUnary) and e.op == '*':
            t = self.static_type(e.e)
            return t.to if isinstance(t, TPtr) else None
        return None

    def is_ptr_expr(self, e):
        if isinstance(e, EName):
            t = self.static_type(e)
            if isinstance(t, (TPtr, TArr)):
                return True
            d = self.find(e.name)
            return d is not None and d.t is None and d.id in self.prov
        if isinstance(e, EUnary) and e.op == '&':
            return True
        if isinstance(e, EBin) and e.op in ('+', '-'):
            return self.is_ptr_expr(e.a) or (e.op == '+' and self.is_ptr_expr(e.b))
        if isinstance(e, ECast):
            return isinstance(e.t, TPtr)
        if isinstance(e, (EIndex, EMember)):
            return isinstance(self.static_type(e), (TPtr, TArr))
        if isinstance(e, ECond):
            return self.is_ptr_expr(e.a) or self.is_ptr_expr(e.b)
        if isinstance(e, EAssign):
            return self.is_ptr_expr(e.l)
        return False

    def prov_of(self, e):
        if isinstance(e, ENum):
            return None
        if isinstance(e, EName):
            d = self.find(e.name)
            if d is not None:
                t = unqual(d.t) if d.t is not None else None
                if isinstance(t, TArr):
                    return self.storage_space(d)
                return self.prov.get(d.id)
            kind, g = self.gen.env.lookup_value(e.name, self.gen.ns)
            if kind == 'global':
                return self.gen.global_space(g)
            if kind == 'const':
                return 'const'
            return 'generic'
        if isinstance(e, EUnary) and e.op == '&':
            return self.addr_space(e.e)
        if isinstance(e, EBin) and e.op in ('+', '-'):
            if self.is_ptr_expr(e.a):
                return self.prov_of(e.a)
            return self.prov_of(e.b)
        if isinstance(e, ECast):
            if isinstance(e.e, (list, EInit)):
                return 'generic'
            if isinstance(e.e, ENum) and e.e.v == 0:
                return None
            if self.is_ptr_expr(e.e):
                return self.prov_of(e.e)
            return 'generic'
        if isinstance(e, ECond):
            return join(self.prov_of(e.a), self.prov_of(e.b))
        if isinstance(e, EAssign):
            return self.prov_of(e.r)
        if isinstance(e, EComma):
            return self.prov_of(e.b)
        if isinstance(e, (EIndex, EMember)):
            t = self.static_type(e)
            if isinstance(t, TArr):
                return self.addr_space(e)
        if isinstance(e, ECall) and isinstance(e.fn, EName) and e.fn.name == 'malloc':
            return 'global'
        return 'generic'

    def storage_space(self, d):
        if '__shared__' in d.storage:
            return 'shared'
        if d.name in self.params and isinstance(d.t, TArr):
            return 'generic'
        return 'local'

    def addr_space(self, e):
        while True:
            if isinstance(e, EName):
                d = self.find(e.name)
                if d is not None:
                    if d.name in self.params and self.params[d.name][2]:
                        return self.params[d.name][1] or 'generic'
                    return self.storage_space(d)
                kind, g = self.gen.env.lookup_value(e.name, self.gen.ns)
                if kind == 'global':
                    return self.gen.global_space(g)
                if kind == 'const':
                    return 'const'
                return 'generic'
            if isinstance(e, EMember):
                if e.arrow:
                    return self.prov_of(e.obj)
                e = e.obj
            elif isinstance(e, EIndex):
                t = self.static_type(e.base)
                if isinstance(t, TArr):
                    e = e.base
                else:
                    return self.prov_of(e.base)
            elif isinstance(e, EUnary) and e.op == '*':
                return self.prov_of(e.e)
            else:
                return 'generic'

    def expr(self, e, ctx='rv'):
        if e is None:
            return
        if isinstance(e, (ENum, EStr, ESizeof, EType)):
            return
        if isinstance(e, EName):
            d = self.find(e.name)
            if d is not None and d.t is not None and isinstance(unqual(d.t), TArr) and ctx == 'rv':
                self.mark_mem(d)     # array decays to a pointer
            return
        if isinstance(e, EMember):
            if e.arrow:
                self.expr(e.obj)
            else:
                self.expr(e.obj, 'lv' if ctx != 'rv' else 'agg')
            if ctx == 'rv' and isinstance(self.static_type(e), TArr):
                self.mark_mem(self.root(e))
            return
        if isinstance(e, EIndex):
            bt = self.static_type(e.base)
            if isinstance(bt, TArr):
                self.expr(e.base, 'agg')
                if const_value(e.idx, self.gen.env, self.cenv) is None:
                    self.mark_mem(self.root(e.base))
            else:
                self.expr(e.base)
            self.expr(e.idx)
            if ctx == 'rv' and isinstance(self.static_type(e), TArr):
                self.mark_mem(self.root(e))
            return
        if isinstance(e, ECall):
            fname = e.fn.name if isinstance(e.fn, EName) else None
            if isinstance(e.fn, EMember):
                self.expr(e.fn.obj, 'agg')
            elif not isinstance(e.fn, EName):
                self.expr(e.fn)
            for i, a in enumerate(e.args):
                if fname in OUT_PARAM_BUILTINS and isinstance(a, EUnary) and a.op == '&':
                    self.expr(a.e, 'lv')
                    self.written.add(getattr(self.root(a.e), 'id', None))
                    continue
                if fname and self.gen.is_user_func(fname) and self.ref_param(fname, len(e.args), i):
                    self.expr(a, 'lv')
                    d = self.root(a)
                    if d is not None:
                        self.written.add(d.id)
                    continue
                self.expr(a)
            return
        if isinstance(e, EUnary):
            if e.op == '&':
                self.expr(e.e, 'lv')
                self.mark_mem(self.root(e.e))
                return
            if e.op in ('++pre', '--pre', '++post', '--post'):
                self.expr(e.e, 'lv')
                d = self.root(e.e)
                if d is not None:
                    self.written.add(d.id)
                return
            self.expr(e.e)
            return
        if isinstance(e, EBin):
            self.expr(e.a)
            self.expr(e.b)
            return
        if isinstance(e, EAssign):
            self.expr(e.l, 'lv')
            self.expr(e.r)
            d = self.root(e.l)
            if d is not None:
                self.written.add(d.id)
                if isinstance(e.l, EName) and (d.t is None or isinstance(unqual(d.t), TPtr)):
                    if e.op == '=':
                        self.set_prov(d, self.prov_of(e.r))
            return
        if isinstance(e, ECond):
            self.expr(e.c)
            self.expr(e.a)
            self.expr(e.b)
            return
        if isinstance(e, ECast):
            if isinstance(e.e, list):
                for x in e.e:
                    self.expr(x)
            else:
                self.expr(e.e)
            return
        if isinstance(e, EComma):
            self.expr(e.a)
            self.expr(e.b, ctx)
            return
        if isinstance(e, EInit):
            for it in e.items:
                if isinstance(it, tuple):
                    self.expr(it[2])
                else:
                    self.expr(it)
            return
        if isinstance(e, EAsm):
            for c, x in e.outs:
                self.expr(x, 'lv')
                d = self.root(x)
                if d is not None:
                    self.written.add(d.id)
            for c, x in e.ins:
                self.expr(x)
            return

    def ref_param(self, fname, nargs, i):
        fds = self.gen.env.lookup_funcs(fname, self.gen.ns) or []
        for fd in fds:
            if len(fd.params) >= nargs and i < len(fd.params) and isinstance(fd.params[i].t, TRef):
                return True
        return False


def name_written(name, node):
    """Is the variable `name` assigned or incremented inside node (ignoring shadowing)?"""
    found = [False]

    def visit(n):
        if found[0] or n is None:
            return
        if isinstance(n, EAssign) and isinstance(n.l, EName) and n.l.name == name:
            found[0] = True
            return
        if isinstance(n, EUnary) and n.op in ('++pre', '--pre', '++post', '--post', '&') and \
                isinstance(n.e, EName) and n.e.name == name:
            found[0] = True
            return
        if isinstance(n, list):
            for x in n:
                visit(x)
            return
        if isinstance(n, tuple):
            for x in n:
                visit(x)
            return
        if hasattr(n, '__slots__'):
            for k in n.__slots__:
                visit(getattr(n, k, None))
        elif isinstance(n, VarDecl):
            visit(n.init)
    visit(node)
    return found[0]


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------
class Frame:
    def __init__(self, name, info, ns):
        self.name = name
        self.info = info
        self.scopes = [{}]
        self.ret_t = None
        self.ret_store = None
        self.ret_label = None
        self.breaks = []
        self.conts = []
        self.ns = ns


SPECIAL_IDX = {'threadIdx': 'tid', 'blockIdx': 'ctaid', 'blockDim': 'ntid', 'gridDim': 'nctaid'}


class Gen:
    def __init__(self, module, env, prelude_funcs):
        self.m = module
        self.env = env
        self.prelude = prelude_funcs
        self.fn = None
        self.frames = []
        self.ns = []
        self.inline_stack = []

    # ---- helpers ------------------------------------------------------------
    @property
    def fr(self):
        return self.frames[-1]

    def err(self, msg):
        raise CompileError(msg)

    def emit(self, s):
        self.fn.emit(s)

    def reg(self, t):
        if t is BOOL:
            return self.fn.reg('r')
        return self.fn.reg(reg_class(t))

    def preg(self):
        return self.fn.reg('p')

    def is_user_func(self, name):
        return self.env.lookup_funcs(name, self.ns) is not None

    def global_space(self, g):
        if '__shared__' in g.storage:
            return 'shared'
        if '__constant__' in g.storage or g in self.env.consts.values():
            return 'const'
        return 'global'

    def lookup(self, name):
        for fr_scope in reversed(self.fr.scopes):
            if name in fr_scope:
                return fr_scope[name]
        return None

    def bind(self, name, b):
        self.fr.scopes[-1][name] = b

    def operand(self, v, t=None):
        """Register name or immediate text of a scalar value."""
        t = t or v.t
        if v.is_imm:
            return fmt_imm(v.imm, t)
        return v.r

    def in_reg(self, v):
        """Value in a register (materialising constants)."""
        if not v.is_imm:
            if v.is_pred:
                r = self.fn.reg('r')
                self.emit('selp.u32 %s, 1, 0, %s;' % (r, v.r))
                return V(v.t, r, space=v.space)
            return v
        t = v.t
        if t is BOOL:
            r = self.fn.reg('r')
            self.emit('mov.u32 %s, %d;' % (r, 1 if v.imm else 0))
            return V(t, r)
        r = self.reg(t)
        self.emit('mov.%s %s, %s;' % (mov_type(t), r, fmt_imm(v.imm, t)))
        return V(t, r, space=v.space)

    def to_pred(self, v):
        if v.is_pred:
            return v.r
        p = self.preg()
        if v.is_imm:
            self.emit('setp.ne.u32 %s, %d, 0;' % (p, 1 if v.imm else 0))
            return p
        t = v.t
        if t.is_float():
            self.emit('setp.neu.%s %s, %s, %s;' % (op_type(t), p, v.r, fmt_imm(0.0, t)))
        else:
            ot = 'u64' if reg_class(t) == 'rd' else 'u32'
            self.emit('setp.ne.%s %s, %s, 0;' % (ot, p, v.r))
        return p

    # ---- conversions ----------------------------------------------------------
    def convert(self, v, t):
        t = unqual(t)
        if isinstance(v, (A, GroupV)):
            if isinstance(v, A) and isinstance(t, TStruct) and v.t is t:
                return v
            if isinstance(v, GroupV):
                return v
            self.err('cannot convert %r to %r' % (v.t, t))
        f = v.t
        if f is t:
            return v
        if isinstance(t, TVoid):
            return V(VOID, imm=0)
        if isinstance(t, (TStruct, TArr)):
            self.err('cannot convert %r to %r' % (f, t))
        if t is BOOL:
            if v.is_imm:
                return V(BOOL, imm=1 if v.imm else 0)
            if f is BOOL:
                return v
            return V(BOOL, self.to_pred(v))
        if v.is_imm:
            if isinstance(t, TPtr):
                return V(t, imm=int(v.imm) & ((1 << 64) - 1), space=v.space)
            if isinstance(f, TPtr):
                return V(t, imm=_wrap(int(v.imm), t))
            val = v.imm
            if t.is_int() and isinstance(val, float):
                if val != val:
                    val = 0
                val = int(max(min(val, 2 ** 63), -2 ** 63))
            return V(t, imm=_wrap(val, t))
        if v.is_pred:
            v = self.in_reg(v)
            f = INT
        if isinstance(t, TPtr):
            if isinstance(f, TPtr):
                return V(t, v.r, space=v.space)
            if f.size == 8:
                return V(t, v.r, space='generic')
            r = self.fn.reg('rd')
            self.emit('cvt.%s.%s %s, %s;' % ('s64' if f.signed else 'u64', 's32' if f.signed else 'u32', r, v.r))
            return V(t, r, space='generic')
        if isinstance(f, TPtr):
            f = ULONG
        if f.is_int() and t.is_int():
            rf, rt = reg_class(f), reg_class(t)
            if rf == 'r' and rt == 'r':
                if t.size >= 4 or t.size >= f.size and (t.signed == f.signed or t.size > f.size):
                    if t.size >= 4:
                        return V(t, v.r)
                r = self.fn.reg('r')
                self.emit('cvt.%s32.%s%d %s, %s;' % ('s' if t.signed else 'u', 's' if t.signed else 'u',
                                                     t.size * 8, r, v.r))
                return V(t, r)
            if rf == 'r' and rt == 'rd':
                r = self.fn.reg('rd')
                self.emit('cvt.%s.%s %s, %s;' % ('s64' if f.signed else 'u64', 's32' if f.signed else 'u32', r, v.r))
                return V(t, r)
            if rf == 'rd' and rt == 'r':
                r = self.fn.reg('r')
                self.emit('cvt.u32.u64 %s, %s;' % (r, v.r))
                if t.size < 4:
                    return self.convert(V(UINT if not t.signed else INT, r), t)
                return V(t, r)
            return V(t, v.r)
        if f.is_int() and t.is_float():
            r = self.reg(t)
            src = 's%d' % max(32, f.size * 8) if f.signed else 'u%d' % max(32, f.size * 8)
            self.emit('cvt.rn.%s.%s %s, %s;' % (op_type(t), src, r, v.r))
            return V(t, r)
        if f.is_float() and t.is_int():
            if t.size < 4:
                tmp = self.convert(v, INT)
                return self.convert(tmp, t)
            r = self.reg(t)
            dst = ('s' if t.signed else 'u') + str(t.size * 8)
            self.emit('cvt.rzi.%s.%s %s, %s;' % (dst, op_type(f), r, v.r))
            return V(t, r)
        if f.is_float() and t.is_float():
            r = self.reg(t)
            if t.size > f.size:
                self.emit('cvt.f64.f32 %s, %s;' % (r, v.r))
            else:
                self.emit('cvt.rn.f32.f64 %s, %s;' % (r, v.r))
            return V(t, r)
        self.err('cannot convert %r to %r' % (f, t))

    # ---- memory ---------------------------------------------------------------
    def addr_text(self, lv, extra=0):
        off = lv.off + extra
        if off:
            return '[%s+%d]' % (lv.base, off)
        return '[%s]' % lv.base

    def space_sfx(self, space):
        return '' if space == 'generic' else '.' + space

    def load(self, lv):
        if isinstance(lv, LReg):
            t = lv.t
            if isinstance(t, (TStruct, TArr)):
                root = lv.store.get(('__type__',))
                n = len(lv.path)
                leaves = {}
                for p, r in lv.store.items():
                    if p == ('__type__',) or p[:n] != lv.path:
                        continue
                    leaves[p[n:]] = r if isinstance(r, V) else V(sub_type(root, p), r)
                return A(t, leaves)
            r = lv.store[lv.path]
            if isinstance(r, V):
                return r
            return V(t, r, space=lv.space)
        if isinstance(lv, GroupV):
            return lv
        t = lv.t
        if isinstance(t, (TStruct, TArr)):
            if not small_agg(t):
                self.err('copying a large aggregate (%d bytes) is not supported' % t.size)
            leaves = {}
            for path, st, off in leaves_of(t):
                leaves[path] = self.load_scalar(lv, st, off)
            return A(t, leaves)
        return self.load_scalar(lv, t, 0)

    def reg_root_type(self, lv):
        return lv.store.get(('__type__',))

    def load_scalar(self, lv, t, extra):
        if isinstance(t, TGroup):
            self.err('group handle in memory')
        r = self.reg(t)
        self.emit('ld%s.%s %s, %s;' % (self.space_sfx(lv.space), mem_type(t), r, self.addr_text(lv, extra)))
        return V(t, r, space='generic' if isinstance(t, TPtr) else None)

    def store(self, lv, v):
        if isinstance(lv, LReg):
            if isinstance(lv.t, (TStruct, TArr)):
                if not isinstance(v, A):
                    self.err('bad aggregate assignment')
                for path, val in v.leaves.items():
                    self.mov_to(lv.store[lv.path + path], val)
                return
            dst = lv.store[lv.path]
            if isinstance(dst, V):
                self.err('assignment to a read-only value')
            self.mov_to(dst, v)
            return
        if getattr(lv, 'readonly', False):
            self.err('assignment to a read-only object')
        if lv.space == 'const':
            self.err('assignment to __constant__ memory')
        t = lv.t
        if isinstance(t, (TStruct, TArr)):
            if not isinstance(v, A):
                self.err('bad aggregate assignment')
            offs = dict((p, (st, o)) for p, st, o in leaves_of(t))
            for path, val in v.leaves.items():
                st, o = offs[path]
                self.store_scalar(lv, st, o, val)
            return
        self.store_scalar(lv, t, 0, v)

    def store_scalar(self, lv, t, extra, v):
        v = self.in_reg(self.convert(v, t))
        self.emit('st%s.%s %s, %s;' % (self.space_sfx(lv.space), mem_type(t), self.addr_text(lv, extra), v.r))

    def mov_to(self, dst, v):
        if isinstance(v, A):
            self.err('aggregate into scalar')
        if dst.startswith('%p'):
            p = self.to_pred(v)
            if p != dst:
                self.emit('mov.pred %s, %s;' % (dst, p))
            return
        t = v.t
        if v.is_pred:
            self.emit('selp.u32 %s, 1, 0, %s;' % (dst, v.r))
            return
        if v.is_imm:
            cls = dst[1:].rstrip('0123456789')
            mt = {'r': 'b32', 'rd': 'b64', 'f': 'f32', 'fd': 'f64'}[cls]
            tt = {'r': UINT, 'rd': ULONG, 'f': FLOAT, 'fd': DOUBLE}[cls]
            val = v.imm
            if cls in ('f', 'fd'):
                val = float(val)
            else:
                val = int(val) if not isinstance(val, float) else int(val)
            self.emit('mov.%s %s, %s;' % (mt, dst, fmt_imm(val, tt)))
            return
        if v.r != dst:
            cls = dst[1:].rstrip('0123456789')
            self.emit('mov.%s %s, %s;' % ({'r': 'b32', 'rd': 'b64', 'f': 'f32', 'fd': 'f64'}[cls], dst, v.r))

    def new_var(self, t, decl=None, space=None):
        """Storage for a local variable of type t: LReg or LMem."""
        info = self.fr.info
        t = unqual(t)
        need_mem = decl is not None and info is not None and decl.id in info.mem
        if isinstance(t, (TStruct, TArr)) and not small_agg(t):
            need_mem = True
        if need_mem:
            off = self.fn.slot(t.size, t.align)
            return LMem(t, 'local', '__pk_depot', off)
        if isinstance(t, (TStruct, TArr)):
            store = {('__type__',): t}
            for path, st, off in leaves_of(t):
                store[path] = self.reg(st)
            return LReg(t, store, ())
        if isinstance(t, TGroup):
            self.err('group variable')
        return LReg(t, {(): self.reg(t)}, (), space)

    def addr_of(self, lv):
        """Generic address (V of pointer type) of an lvalue in memory."""
        if isinstance(lv, LReg):
            self.err('address of a register variable')
        t = TPtr(lv.t)
        if lv.space == 'generic':
            if lv.off == 0 and lv.base.startswith('%'):
                return V(t, lv.base, space='generic')
            r = self.fn.reg('rd')
            self.emit('add.s64 %s, %s, %d;' % (r, lv.base, lv.off))
            return V(t, r, space='generic')
        if lv.space == 'param':
            self.err('address of a kernel parameter')
        r = self.fn.reg('rd')
        if lv.base.startswith('%'):
            if lv.off:
                self.emit('add.s64 %s, %s, %d;' % (r, lv.base, lv.off))
            else:
                r = lv.base
        else:
            self.emit('mov.u64 %s, %s;' % (r, lv.base))
            if lv.off:
                r2 = self.fn.reg('rd')
                self.emit('add.s64 %s, %s, %d;' % (r2, r, lv.off))
                r = r2
        g = self.fn.reg('rd')
        self.emit('cvta.%s.u64 %s, %s;' % (lv.space, g, r))
        return V(t, g, space=lv.space)

    def deref(self, pv, t=None):
        """LMem for *pv (pv is a pointer value)."""
        t = t or pv.t.to
        if isinstance(t, TVoid):
            self.err('dereferencing void*')
        pv = self.in_reg(pv)
        space = pv.space or 'generic'
        if space == 'generic':
            return LMem(t, 'generic', pv.r)
        r = self.fn.reg('rd')
        self.emit('cvta.to.%s.u64 %s, %s;' % (space, r, pv.r))
        return LMem(t, space, r)

    def lmem_index(self, lv, idx, elem):
        """lv is an array in memory, idx a scalar V: LMem of lv[idx]."""
        if idx.is_imm:
            return LMem(elem, lv.space, lv.base, lv.off + int(idx.imm) * elem.size, lv.readonly)
        base = lv.base
        if not base.startswith('%'):
            r = self.fn.reg('rd')
            self.emit('mov.u64 %s, %s;' % (r, base))
            base = r
        off = self.scaled(idx, elem.size)
        r = self.fn.reg('rd')
        self.emit('add.s64 %s, %s, %s;' % (r, base, off))
        return LMem(elem, lv.space, r, lv.off, lv.readonly)

    def scaled(self, idx, size):
        """64-bit byte offset idx*size for an integer index value."""
        t = idx.t
        if t is BOOL or idx.is_pred:
            idx = self.convert(self.in_reg(idx), INT)
            t = INT
        r = self.fn.reg('rd')
        if reg_class(t) == 'r':
            self.emit('mul.wide.%s %s, %s, %d;' % ('s32' if t.signed else 'u32', r, idx.r, size))
        else:
            if size & (size - 1) == 0:
                self.emit('shl.b64 %s, %s, %d;' % (r, idx.r, size.bit_length() - 1))
            else:
                self.emit('mul.lo.s64 %s, %s, %d;' % (r, idx.r, size))
        return r

    # ---- expressions: lvalues ----------------------------------------------------
    def lv(self, e):
        if isinstance(e, EName):
            b = self.resolve(e.name)
            if isinstance(b, (LReg, LMem, GroupV)):
                return b
            if isinstance(b, V):
                return LReg(b.t, {(): b}, (), b.space)
            if isinstance(b, A):
                store = {('__type__',): b.t}
                for p, v in b.leaves.items():
                    store[p] = v
                return LReg(b.t, store, ())
            self.err("'%s' is not an lvalue" % e.name)
        if isinstance(e, EMember):
            if e.arrow:
                pv = self.rv(e.obj)
                if not isinstance(pv, V) or not isinstance(pv.t, TPtr):
                    self.err('-> on a non-pointer')
                base = self.deref(pv)
            else:
                if isinstance(e.obj, EName) and e.obj.name in SPECIAL_IDX and self.lookup(e.obj.name) is None:
                    raise NotLvalue('built-in index variables are read-only')
                base = self.lv(e.obj)
            st = unqual(base.t)
            if not isinstance(st, TStruct):
                self.err('member access on %r' % (st,))
            f = st.field(e.name)
            if f is None:
                self.err("no member '%s' in %r" % (e.name, st))
            if isinstance(base, LReg):
                return LReg(f[1], base.store, base.path + (e.name,))
            return LMem(f[1], base.space, base.base, base.off + f[2], base.readonly)
        if isinstance(e, EIndex):
            bt = self.type_of_lv_base(e.base)
            if isinstance(bt, TArr):
                base = self.lv(e.base)
                idx = self.rv_scalar(e.idx)
                if isinstance(base, LReg):
                    if not idx.is_imm:
                        raise NeedMemory(base)
                    i = int(idx.imm)
                    if i < 0 or i >= bt.n:
                        self.err('constant index %d out of range' % i)
                    return LReg(bt.elem, base.store, base.path + (i,))
                return self.lmem_index(base, idx, bt.elem)
            pv = self.rv(e.base)
            idx = self.rv(e.idx)
            if isinstance(idx, V) and isinstance(idx.t, TPtr):
                pv, idx = idx, pv
            if not isinstance(pv, V) or not isinstance(pv.t, TPtr):
                self.err('subscript of a non-pointer')
            p = self.ptr_add(pv, idx, 1)
            return self.deref(p)
        if isinstance(e, EUnary) and e.op == '*':
            pv = self.rv(e.e)
            if not isinstance(pv, V) or not isinstance(pv.t, TPtr):
                self.err('* on a non-pointer')
            return self.deref(pv)
        if isinstance(e, EComma):
            self.rv(e.a)
            return self.lv(e.b)
        if isinstance(e, EAssign):
            self.rv(e)
            return self.lv(e.l)
        if isinstance(e, EUnary) and e.op in ('++pre', '--pre'):
            self.rv(e)
            return self.lv(e.e)
        if isinstance(e, ECast) and isinstance(e.t, TRef):
            return self.lv(e.e)
        raise NotLvalue('expression is not an lvalue')

    def type_of_lv_base(self, e):
        """Declared type of an expression used as an array base (to tell arrays from pointers)."""
        if isinstance(e, EName):
            b = self.resolve(e.name, quiet=True)
            if isinstance(b, (LReg, LMem, A)):
                return unqual(b.t)
            if isinstance(b, V):
                return b.t
            return None
        if isinstance(e, EMember):
            if e.arrow:
                pt = self.type_of_rv(e.obj)
                st = pt.to if isinstance(pt, TPtr) else None
            else:
                st = self.type_of_lv_base(e.obj)
            if isinstance(st, TStruct):
                f = st.field(e.name)
                return f[1] if f else None
            return None
        if isinstance(e, EIndex):
            bt = self.type_of_lv_base(e.base)
            if isinstance(bt, TArr):
                return bt.elem
            if isinstance(bt, TPtr):
                return bt.to
            return None
        if isinstance(e, EUnary) and e.op == '*':
            pt = self.type_of_rv(e.e)
            return pt.to if isinstance(pt, TPtr) else None
        return None

    def type_of_rv(self, e):
        t = self.type_of_lv_base(e)
        if isinstance(t, TArr):
            return TPtr(t.elem)
        return t

    # ---- name resolution ---------------------------------------------------------
    def resolve(self, name, quiet=False):
        b = self.lookup(name)
        if b is not None:
            return b
        kind, v = self.env.lookup_value(name, self.ns)
        if kind == 'enum':
            return V(v[1], imm=v[0])
        if kind == 'global':
            return self.global_lv(v)
        if kind == 'const':
            if v.t.is_arith() and v.init is not None and not isinstance(v.init, EInit):
                r = const_value(v.init, self.env, {})
                if r is not None:
                    return V(v.t, imm=_wrap(r[0], v.t))
            return self.const_global_lv(v)
        if name == 'warpSize':
            return V(INT, imm=32)
        if quiet:
            return None
        self.err("unknown name '%s'" % name)

    def global_lv(self, g):
        sym = 'pk_g_' + ident(g.name)
        space = self.global_space(g)
        if space == 'shared':
            if 'extern' in g.storage:
                self.m.dynsmem = True
                return LMem(g.t, 'shared', 'pk_dynsmem', 0)
            self.m.declare(sym, '.shared .align %d .b8 %s[%d];' % (max(g.t.align, 4), sym, max(g.t.size, 1)))
            return LMem(g.t, 'shared', sym, 0)
        self.m.declare(sym, '.%s .align %d .b8 %s[%d];' % (space, max(g.t.align, 4), sym, max(g.t.size, 1)))
        return LMem(g.t, space, sym, 0)

    def const_global_lv(self, g):
        """A file-scope const aggregate used in device code: emitted as initialised .const data."""
        sym = 'pk_c_' + ident(g.name)
        if sym not in self.m.declared:
            data = self.const_bytes(g.t, g.init)
            self.m.declare(sym, '.const .align %d .b8 %s[%d] = {%s};' % (max(g.t.align, 4), sym, len(data),
                                                                       ', '.join(map(str, data))))
        return LMem(g.t, 'const', sym, 0, True)

    def const_bytes(self, t, init):
        buf = bytearray(max(t.size, 1))

        def put(t, off, e):
            if e is None:
                return
            if isinstance(t, (TArr, TStruct)):
                items = e.items if isinstance(e, EInit) else [e]
                if isinstance(t, TArr):
                    for i, it in enumerate(items):
                        put(t.elem, off + i * t.elem.size, it)
                else:
                    for (fname, ft, fo), it in zip(t.fields, items):
                        put(ft, off + fo, it)
                return
            if isinstance(e, EInit):
                e = e.items[0] if e.items else None
                if e is None:
                    return
            r = const_value(e, self.env, {})
            if r is None:
                self.err('non-constant initialiser for a constant array')
            v = r[0]
            if t.is_float():
                buf[off:off + t.size] = struct.pack('<f' if t.size == 4 else '<d', float(v))
            else:
                buf[off:off + t.size] = (int(v) & ((1 << (8 * t.size)) - 1)).to_bytes(t.size, 'little')
        put(t, 0, init)
        return list(buf)

    # ---- expressions: rvalues ------------------------------------------------------
    def rv_scalar(self, e):
        v = self.rv(e)
        if not isinstance(v, V):
            self.err('expected a scalar value')
        return v

    def rv(self, e):
        m = getattr(self, 'rv_' + type(e).__name__, None)
        if m is None:
            self.err('unsupported expression %s' % type(e).__name__)
        return m(e)

    def decay(self, lv):
        """Value of an lvalue: arrays decay to pointers."""
        if isinstance(lv, GroupV):
            return lv
        t = unqual(lv.t)
        if isinstance(t, TArr):
            if isinstance(lv, LReg):
                raise NeedMemory(lv)
            p = self.addr_of(LMem(t.elem, lv.space, lv.base, lv.off))
            return p
        return self.load(lv)

    def rv_ENum(self, e):
        return V(e.t, imm=e.v)

    def rv_EStr(self, e):
        sym = self.m.string(e.v)
        r = self.fn.reg('rd')
        self.emit('mov.u64 %s, %s;' % (r, sym))
        g = self.fn.reg('rd')
        self.emit('cvta.global.u64 %s, %s;' % (g, r))
        return V(TPtr(CHAR), g, space='global')

    def rv_EName(self, e):
        if e.name in SPECIAL_IDX and self.lookup(e.name) is None:
            return A(VEC['uint3'], dict(((c,), self.special(e.name, c)) for c in 'xyz'))
        b = self.resolve(e.name)
        if isinstance(b, (V, A, GroupV)):
            return b
        return self.decay(b)

    def special(self, name, comp):
        r = self.fn.reg('r')
        self.emit('mov.u32 %s, %%%s.%s;' % (r, SPECIAL_IDX[name], comp))
        return V(UINT, r)

    def rv_EMember(self, e):
        if isinstance(e.obj, EName) and e.obj.name in SPECIAL_IDX and self.lookup(e.obj.name) is None:
            if e.name not in ('x', 'y', 'z'):
                self.err('bad member of %s' % e.obj.name)
            return self.special(e.obj.name, e.name)
        try:
            return self.decay(self.lv(e))
        except NotLvalue:
            pass
        obj = self.rv(e.obj)
        if isinstance(obj, A):
            st = obj.t
            f = st.field(e.name)
            if f is None:
                self.err("no member '%s'" % e.name)
            sub = dict((p[1:], v) for p, v in obj.leaves.items() if p[0] == e.name)
            if isinstance(f[1], (TStruct, TArr)):
                return A(f[1], sub)
            return sub[()]
        self.err('member access on a non-struct value')

    def rv_EIndex(self, e):
        return self.decay(self.lv(e))

    def rv_EComma(self, e):
        self.rv(e.a)
        return self.rv(e.b)

    def rv_ESizeof(self, e):
        if e.t is not None:
            return V(ULONG, imm=e.t.size)
        t = self.type_of_lv_base(e.e)
        if t is None:
            save = self.fn.lines[:]
            v = self.rv(e.e)
            self.fn.lines[:] = save
            t = v.t
        return V(ULONG, imm=t.size)

    def rv_EType(self, e):
        return GroupV(e.t)

    def rv_ECast(self, e):
        t = unqual(e.t)
        if isinstance(e.e, EInit):
            if isinstance(t, (TStruct, TArr)):
                return self.agg_init_value(t, e.e)
            items = e.e.items
            if not items:
                return V(t, imm=0) if t.is_arith() else self.err('empty initialiser')
            return self.convert(self.rv(items[0]), t)
        v = self.rv(e.e)
        if isinstance(t, TStruct) and isinstance(v, V) and t.vec is not None:
            # e.g. float4(x) is not valid CUDA, but make_float4-like broadcasts are common in helpers
            self.err('cannot convert a scalar to %r' % (t,))
        if isinstance(t, TPtr) and isinstance(v, V) and isinstance(v.t, TPtr):
            return V(t, v.r, v.imm, v.space)
        return self.convert(v, t)

    def rv_EInit(self, e):
        self.err('brace initialiser in this context')

    def rv_EUnary(self, e):
        op = e.op
        if op == '&':
            if isinstance(e.e, EName) and self.is_user_func(e.e.name):
                self.err('device function pointers are not supported')
            lv = self.lv(e.e)
            return self.addr_of(lv)
        if op == '*':
            return self.decay(self.lv(e))
        if op in ('++pre', '--pre', '++post', '--post'):
            lv = self.lv(e.e)
            old = self.load(lv)
            if not isinstance(old, V):
                self.err('++/-- on an aggregate')
            if op.endswith('post') and not old.is_imm:
                keep = self.reg(old.t) if not old.is_pred else self.fn.reg('r')
                self.mov_to(keep, old)
                old_copy = V(old.t, keep, space=old.space)
            else:
                old_copy = None
            one = V(INT, imm=1)
            if isinstance(old.t, TPtr):
                new = self.ptr_add(old, one, 1 if op[0] == '+' else -1)
            else:
                t = old.t
                ct = promote(t)
                new = self.arith('+' if op[0] == '+' else '-', self.convert(old, ct), self.convert(one, ct))
                new = self.convert(new, t)
            self.store(lv, new)
            if op.endswith('post'):
                return old_copy if old_copy is not None else old
            return new
        v = self.rv(e.e)
        if not isinstance(v, V):
            self.err('unary %s on an aggregate' % op)
        if op == '!':
            if v.is_imm:
                return V(BOOL, imm=0 if v.imm else 1)
            p = self.to_pred(v)
            q = self.preg()
            self.emit('not.pred %s, %s;' % (q, p))
            return V(BOOL, q)
        if op == '+':
            return self.convert(v, promote(v.t)) if v.t.is_int() else v
        if op == '-':
            t = promote(v.t) if v.t.is_int() else v.t
            v = self.convert(v, t)
            if v.is_imm:
                return V(t, imm=_wrap(-v.imm, t))
            r = self.reg(t)
            self.emit('neg.%s %s, %s;' % (op_type(t) if not t.is_int() else ('s64' if t.size == 8 else 's32'), r,
                                          v.r))
            return V(t, r)
        if op == '~':
            t = promote(v.t)
            v = self.convert(v, t)
            if v.is_imm:
                return V(t, imm=_wrap(~int(v.imm), t))
            r = self.reg(t)
            self.emit('not.b%d %s, %s;' % (64 if t.size == 8 else 32, r, v.r))
            return V(t, r)
        self.err('unary %s' % op)

    def ptr_add(self, pv, n, sign):
        """pointer + sign*n."""
        el = pv.t.to
        size = 1 if isinstance(el, TVoid) else el.size
        n = self.convert(n, n.t if n.t.is_int() and not n.is_pred else INT) if isinstance(n, V) else n
        if n.is_imm and pv.is_imm:
            return V(pv.t, imm=(pv.imm + sign * int(n.imm) * size) & ((1 << 64) - 1), space=pv.space)
        if n.is_imm:
            if n.imm == 0:
                return pv
            pv = self.in_reg(pv)
            r = self.fn.reg('rd')
            self.emit('add.s64 %s, %s, %d;' % (r, pv.r, sign * int(n.imm) * size))
            return V(pv.t, r, space=pv.space)
        pv = self.in_reg(pv)
        off = self.scaled(n, size)
        r = self.fn.reg('rd')
        self.emit('%s.s64 %s, %s, %s;' % ('add' if sign > 0 else 'sub', r, pv.r, off))
        return V(pv.t, r, space=pv.space)

    def arith(self, op, a, b):
        """a op b for arithmetic operands already converted to their common type."""
        t = a.t
        if a.is_imm and b.is_imm:
            r = fold_bin(op, (a.imm, t), (b.imm, b.t))
            if r is not None:
                return V(r[1], imm=r[0])
        ot = op_type(t)
        r = self.reg(t)
        A_, B_ = self.operand(self.in_reg(a) if a.is_imm and op in ('/', '%') else a), self.operand(b)
        if a.is_imm:
            A_ = self.in_reg(a).r
        if op == '+':
            self.emit('add.%s %s, %s, %s;' % (ot, r, A_, B_))
        elif op == '-':
            self.emit('sub.%s %s, %s, %s;' % (ot, r, A_, B_))
        elif op == '*':
            if t.is_float():
                self.emit('mul.%s %s, %s, %s;' % (ot, r, A_, B_))
            else:
                self.emit('mul.lo.%s %s, %s, %s;' % (ot, r, A_, B_))
        elif op == '/':
            if t.is_float():
                self.emit('div.rn.%s %s, %s, %s;' % (ot, r, A_, B_))
            else:
                self.emit('div.%s %s, %s, %s;' % (ot, r, A_, B_))
        elif op == '%':
            if t.is_float():
                self.err('% on floating point')
            self.emit('rem.%s %s, %s, %s;' % (ot, r, A_, B_))
        elif op in ('&', '|', '^'):
            if t.is_float():
                self.err('bitwise operator on floating point')
            name = {'&': 'and', '|': 'or', '^': 'xor'}[op]
            self.emit('%s.b%d %s, %s, %s;' % (name, 64 if t.size == 8 else 32, r, A_, B_))
        else:
            self.err('operator %s' % op)
        return V(t, r)

    def compare(self, op, a, b):
        if isinstance(a.t, TPtr) or isinstance(b.t, TPtr):
            t = ULONG
            a = self.convert(a, t) if not isinstance(a.t, TPtr) else V(t, a.r, a.imm)
            b = self.convert(b, t) if not isinstance(b.t, TPtr) else V(t, b.r, b.imm)
        else:
            t = common_type(a.t, b.t)
            a, b = self.convert(a, t), self.convert(b, t)
        if a.is_imm and b.is_imm:
            r = fold_bin(op, (a.imm, t), (b.imm, t))
            return V(BOOL, imm=r[0])
        cmp = {'==': 'eq', '!=': 'ne', '<': 'lt', '>': 'gt', '<=': 'le', '>=': 'ge'}[op]
        if t.is_float() and op == '!=':
            cmp = 'neu'
        if a.is_imm:
            a = self.in_reg(a)
        p = self.preg()
        self.emit('setp.%s.%s %s, %s, %s;' % (cmp, op_type(t), p, a.r, self.operand(b)))
        return V(BOOL, p)

    def rv_EBin(self, e):
        op = e.op
        if op in ('&&', '||'):
            return self.logical(e)
        a = self.rv(e.a)
        b = self.rv(e.b)
        if not isinstance(a, V) or not isinstance(b, V):
            return self.agg_binop(op, a, b)
        if op in ('==', '!=', '<', '>', '<=', '>='):
            return self.compare(op, a, b)
        if op in ('+', '-') and (isinstance(a.t, TPtr) or isinstance(b.t, TPtr)):
            if isinstance(a.t, TPtr) and isinstance(b.t, TPtr):
                if op != '-':
                    self.err('pointer + pointer')
                da, db = self.in_reg(a), self.in_reg(b)
                r = self.fn.reg('rd')
                self.emit('sub.s64 %s, %s, %s;' % (r, da.r, db.r))
                size = max(1, a.t.to.size if not isinstance(a.t.to, TVoid) else 1)
                if size > 1:
                    r2 = self.fn.reg('rd')
                    self.emit('div.s64 %s, %s, %d;' % (r2, r, size))
                    r = r2
                return V(LONG, r)
            if isinstance(a.t, TPtr):
                return self.ptr_add(a, b, 1 if op == '+' else -1)
            return self.ptr_add(b, a, 1)
        if op in ('<<', '>>'):
            t = promote(a.t)
            a = self.convert(a, t)
            b = self.convert(b, promote(b.t))
            if a.is_imm and b.is_imm:
                r = fold_bin(op, (a.imm, t), (b.imm, b.t))
                if r is not None:
                    return V(t, imm=r[0])
            a = self.in_reg(a)
            cnt = self.operand(b) if b.is_imm else self.convert(b, UINT).r
            if b.is_imm:
                cnt = str(int(b.imm) & 0xffffffff)
            r = self.reg(t)
            bits = 64 if t.size == 8 else 32
            if op == '<<':
                self.emit('shl.b%d %s, %s, %s;' % (bits, r, a.r, cnt))
            else:
                self.emit('shr.%s%d %s, %s, %s;' % ('s' if t.signed else 'u', bits, r, a.r, cnt))
            return V(t, r)
        t = common_type(a.t, b.t)
        return self.arith(op, self.convert(a, t), self.convert(b, t))

    def agg_binop(self, op, a, b):
        self.err('operator %s on aggregates (operator overloading is not supported)' % op)

    def logical(self, e):
        a = self.rv(e.a)
        if isinstance(a, V) and a.is_imm:
            if (e.op == '&&' and not a.imm) or (e.op == '||' and a.imm):
                return V(BOOL, imm=1 if a.imm else 0)
            b = self.rv(e.b)
            return self.convert(b, BOOL)
        res = self.preg()
        pa = self.to_pred(a)
        self.emit('mov.pred %s, %s;' % (res, pa))
        done = self.fn.label()
        if e.op == '&&':
            self.emit('@!%s bra %s;' % (res, done))
        else:
            self.emit('@%s bra %s;' % (res, done))
        b = self.rv(e.b)
        pb = self.to_pred(b) if not (isinstance(b, V) and b.is_imm) else None
        if pb is None:
            self.emit('setp.ne.u32 %s, %d, 0;' % (res, 1 if b.imm else 0))
        else:
            self.emit('mov.pred %s, %s;' % (res, pb))
        self.fn.place(done)
        return V(BOOL, res)

    def simple(self, e):
        """Side-effect free and cheap enough to evaluate unconditionally."""
        if isinstance(e, (ENum, EName)):
            return True
        if isinstance(e, EMember):
            return not e.arrow and self.simple(e.obj)
        if isinstance(e, ECast) and not isinstance(e.e, (list, EInit)):
            return self.simple(e.e)
        if isinstance(e, EUnary) and e.op in ('-', '+', '!', '~'):
            return self.simple(e.e)
        if isinstance(e, EBin) and e.op not in ('/', '%', '&&', '||'):
            return self.simple(e.a) and self.simple(e.b)
        return False

    def rv_ECond(self, e):
        c = self.rv(e.c)
        if isinstance(c, V) and c.is_imm:
            return self.rv(e.a if c.imm else e.b)
        if self.simple(e.a) and self.simple(e.b):
            p = self.to_pred(c)
            a = self.rv(e.a)
            b = self.rv(e.b)
            if isinstance(a, V) and isinstance(b, V):
                t = self.cond_type(a, b)
                a, b = self.convert(a, t), self.convert(b, t)
                if t is BOOL:
                    a, b = self.in_reg(a), self.in_reg(b)
                r = self.reg(t)
                if a.is_imm and b.is_imm:
                    pass
                st = 'b64' if reg_class(t) == 'rd' else ('f32' if reg_class(t) == 'f' else
                                                        ('f64' if reg_class(t) == 'fd' else 'b32'))
                self.emit('selp.%s %s, %s, %s, %s;' % (st, r, self.operand(a), self.operand(b), p))
                return V(t, r, space=join(a.space, b.space) if isinstance(t, TPtr) else None)
        p = self.to_pred(c)
        lab_else, lab_end = self.fn.label(), self.fn.label()
        self.emit('@!%s bra %s;' % (p, lab_else))
        a = self.rv(e.a)
        if isinstance(a, A):
            store = self.new_var(a.t)
            self.store(store, a)
            self.emit('bra.uni %s;' % lab_end)
            self.fn.place(lab_else)
            b = self.convert(self.rv(e.b), a.t)
            self.store(store, b)
            self.fn.place(lab_end)
            return self.load(store)
        # the type is decided by both branches: evaluate the else branch's type without code
        tb = self.peek_type(e.b)
        t = self.cond_type(a, V(tb) if tb is not None else a)
        res = self.reg(t)
        self.mov_to(res, self.convert(a, t))
        self.emit('bra.uni %s;' % lab_end)
        self.fn.place(lab_else)
        b = self.rv(e.b)
        self.mov_to(res, self.convert(b, t))
        self.fn.place(lab_end)
        sp = None
        if isinstance(t, TPtr):
            sp = join(a.space, b.space if isinstance(b, V) else None) or 'generic'
        return V(t, res, space=sp)

    def peek_type(self, e):
        save_lines = self.fn.lines[:]
        save_n = dict(self.fn.n)
        save_lab = self.fn.nlabel
        try:
            v = self.rv(e)
            return v.t
        except CompileError:
            return None
        finally:
            self.fn.lines[:] = save_lines
            self.fn.n = save_n
            self.fn.nlabel = save_lab

    def cond_type(self, a, b):
        if isinstance(a.t, TPtr):
            return a.t
        if isinstance(b.t, TPtr):
            return b.t
        if a.t is BOOL and b.t is BOOL:
            return BOOL
        return common_type(a.t, b.t)

    def rv_EAssign(self, e):
        lv = self.lv(e.l)
        t = unqual(lv.t)
        if e.op == '=':
            if isinstance(e.r, EInit):
                v = self.agg_init_value(t, e.r) if isinstance(t, (TStruct, TArr)) else \
                    self.convert(self.rv(e.r.items[0]) if e.r.items else V(INT, imm=0), t)
            else:
                v = self.rv(e.r)
                v = self.convert(v, t) if not isinstance(t, TGroup) else v
            self.store(lv, v)
            return v
        op = e.op[:-1]
        cur = self.load(lv)
        r = self.rv(e.r)
        if isinstance(cur.t, TPtr):
            if op not in ('+', '-'):
                self.err('bad pointer compound assignment')
            new = self.ptr_add(cur, r, 1 if op == '+' else -1)
        elif op in ('<<', '>>'):
            new = self.rv_EBin_values(op, cur, r)
        else:
            ct = common_type(cur.t, r.t)
            new = self.arith(op, self.convert(cur, ct), self.convert(r, ct))
        new = self.convert(new, t)
        self.store(lv, new)
        return new

    def rv_EBin_values(self, op, a, b):
        e = EBin(op, None, None)
        saved = self.rv
        vals = iter([a, b])
        self.rv = lambda x: next(vals)
        try:
            return self.rv_EBin(e)
        finally:
            self.rv = saved

    def agg_init_value(self, t, init):
        """Value of an aggregate initialiser {..} for type t (missing members are zero)."""
        leaves = {}
        for path, st, off in leaves_of(t):
            leaves[path] = V(st, imm=0 if not st.is_float() else 0.0)

        def put(t, prefix, e):
            if isinstance(t, (TStruct, TArr)):
                if not isinstance(e, EInit):
                    v = self.rv(e)
                    if isinstance(v, A) and v.t is t:
                        for p, x in v.leaves.items():
                            leaves[prefix + p] = x
                        return
                    if isinstance(t, TStruct) and t.vec is None:
                        self.err('bad initialiser for %r' % (t,))
                    items = [e]
                else:
                    items = e.items
                if isinstance(t, TArr):
                    for i, it in enumerate(items):
                        put(t.elem, prefix + (i,), it)
                else:
                    pos = 0
                    for it in items:
                        if isinstance(it, tuple) and it[0] == '.':
                            f = t.field(it[1])
                            put(f[1], prefix + (it[1],), it[2])
                            pos = t.fields.index(f) + 1
                            continue
                        if pos >= len(t.fields):
                            self.err('too many initialisers')
                        fname, ft, fo = t.fields[pos]
                        put(ft, prefix + (fname,), it)
                        pos += 1
                return
            if isinstance(e, EInit):
                e = e.items[0] if e.items else ENum(0, INT)
            leaves[prefix] = self.convert(self.rv(e), t)
        put(t, (), init)
        return A(t, leaves)

    # ---- calls ---------------------------------------------------------------------
    def rv_ECall(self, e):
        fn = e.fn
        if isinstance(fn, EMember):
            return self.method_call(fn, e.args)
        if not isinstance(fn, EName):
            self.err('indirect calls are not supported')
        name = fn.name
        last = name.split('::')[-1]
        if self.lookup(name) is not None:
            self.err("'%s' is not a function" % name)
        fds = self.env.lookup_funcs(name, self.ns)
        user = [fd for fd in (fds or []) if fd.kind != 'host']
        if user:
            fd = self.resolve_overload(user, e.args, name)
            if fd is not None:
                return self.inline(fd, e.args)
        h = BUILTINS.get(last)
        if h is not None:
            return h(self, e.args, fn.targs)
        pre = self.prelude.get(last)
        if pre:
            fd = self.resolve_overload(pre, e.args, last)
            if fd is not None:
                return self.inline(fd, e.args)
        if fds:
            self.err("calling host function '%s' from device code" % name)
        self.err("unknown function '%s'" % name)

    def arg_types(self, args):
        out = []
        for a in args:
            t = self.peek_type(a)
            out.append(t)
        return out

    def resolve_overload(self, fds, args, name):
        cands = []
        for fd in fds:
            n = len(fd.params)
            req = len([p for p in fd.params if p.init is None])
            if req <= len(args) <= n:
                cands.append(fd)
        if not cands:
            return None
        defs = [fd for fd in cands if fd.body is not None] or cands
        if len(defs) == 1:
            return self.definition(defs[0], name)
        types = self.arg_types(args)
        best, best_score = None, None
        for fd in defs:
            score = 0
            for p, at in zip(fd.params, types):
                pt = unqual(p.t)
                if at is None:
                    score += 1
                elif pt is at or (isinstance(pt, TPtr) and isinstance(at, TPtr) and
                                  (pt.to is at.to or isinstance(pt.to, TVoid))):
                    score += 0
                elif pt.is_arith() and at.is_arith():
                    if pt.is_float() == at.is_float():
                        score += 1 if promote(at) is pt else 2
                    else:
                        score += 3
                elif isinstance(pt, TStruct) and pt is at:
                    score += 0
                else:
                    score += 10
            if best_score is None or score < best_score:
                best, best_score = fd, score
        return self.definition(best, name)

    def definition(self, fd, name):
        if fd.body is not None:
            return fd
        for other in self.env.lookup_funcs(name, self.ns) or []:
            if other.body is not None and len(other.params) == len(fd.params) and \
                    all(repr(a.t) == repr(b.t) for a, b in zip(other.params, fd.params)):
                return other
        self.err("device function '%s' has no definition in this file" % name)

    def inline(self, fd, args):
        key = (fd.name, id(fd))
        if key in self.inline_stack:
            self.err("recursive device function '%s' is not supported" % fd.name)
        if len(self.inline_stack) > 40:
            self.err('device functions nested too deeply')
        params = {}
        binds = []
        for i, p in enumerate(fd.params):
            arg = args[i] if i < len(args) else p.init
            pt = p.t
            if isinstance(pt, TRef):
                try:
                    lv = self.lv(arg)
                    if isinstance(lv, LMem) or isinstance(lv, LReg):
                        prov = None
                        if isinstance(lv, LMem):
                            prov = lv.space
                        binds.append((p, 'ref', lv))
                        params[p.name] = (p, prov, True)
                        continue
                except (CompileError, NotLvalue):
                    pass
                v = self.rv(arg)
                v = self.convert(v, unqual(pt)) if not isinstance(v, (A, GroupV)) else v
                binds.append((p, 'val', v))
                params[p.name] = (p, getattr(v, 'space', None), False)
                continue
            if isinstance(pt, TGroup):
                v = self.rv(arg)
                binds.append((p, 'val', v))
                params[p.name] = (p, None, False)
                continue
            if isinstance(pt, TFunctor):
                binds.append((p, 'val', GroupV(pt)))
                params[p.name] = (p, None, False)
                continue
            v = self.rv(arg)
            if isinstance(v, GroupV):
                binds.append((p, 'val', v))
                params[p.name] = (p, None, False)
                continue
            v = self.convert(v, pt)
            binds.append((p, 'val', v))
            params[p.name] = (p, v.space if isinstance(v, V) else None, False)
        info = Analysis(self, params)
        saved_ns = self.ns
        self.ns = list(fd.ns or [])
        info.run(fd.body)
        frame = Frame(fd.name, info, self.ns)
        self.frames.append(frame)
        self.inline_stack.append(key)
        try:
            for p, kind, v in binds:
                if p.name is None:
                    continue
                if kind == 'ref':
                    self.bind(p.name, v)
                    continue
                if isinstance(v, GroupV):
                    self.bind(p.name, v)
                    continue
                if p.id in info.written or p.id in info.mem or (isinstance(v, V) and v.is_pred):
                    st = self.new_var(p.t, p, space=v.space if isinstance(v, V) else None)
                    self.store(st, v)
                    if isinstance(st, LReg) and isinstance(p.t, TPtr):
                        st.space = info.prov.get(p.id, v.space)
                    self.bind(p.name, st)
                else:
                    self.bind(p.name, v)
            rt = unqual(fd.ret)
            frame.ret_t = rt
            frame.ret_label = self.fn.label()
            if not isinstance(rt, TVoid):
                if isinstance(rt, TGroup):
                    self.err('functions returning groups are not supported')
                frame.ret_store = self.new_var(rt)
            self.block_body(fd.body)
            self.fn.place(frame.ret_label)
            if frame.ret_store is not None:
                v = self.load(frame.ret_store)
                if isinstance(v, V) and isinstance(rt, TPtr):
                    v.space = getattr(frame, 'ret_space', None) or 'generic'
                return v
            return V(VOID, imm=0)
        finally:
            self.inline_stack.pop()
            self.frames.pop()
            self.ns = saved_ns

    def method_call(self, fn, args):
        obj = self.rv(fn.obj)
        name = fn.name
        targs = None
        if isinstance(name, tuple):
            name, targs = name
        if isinstance(obj, GroupV):
            return group_method(self, obj, name, args, targs)
        self.err('member function calls are not supported (%s)' % name)

    # ---- statements -----------------------------------------------------------------
    def block_body(self, blk):
        self.fr.scopes.append({})
        try:
            for s in blk.stmts:
                self.stmt(s)
        finally:
            self.fr.scopes.pop()

    def stmt(self, s):
        m = getattr(self, 'st_' + type(s).__name__, None)
        if m is None:
            self.err('unsupported statement %s' % type(s).__name__)
        m(s)

    def st_SEmpty(self, s):
        pass

    def st_SBlock(self, s):
        self.block_body(s)

    def st_SExpr(self, s):
        self.rv(s.e)

    def st_SDecl(self, s):
        for d in s.decls:
            self.declare_local(d)

    def declare_local(self, d):
        t = d.t
        if '__shared__' in d.storage:
            if t is None:
                self.err('auto __shared__')
            if 'extern' in d.storage:
                self.m.dynsmem = True
                self.bind(d.name, LMem(t, 'shared', 'pk_dynsmem', 0))
                return
            sym = 'pk_sh_%s_%s' % (ident(self.fr.name), ident(d.name))
            self.m.declare(sym, '.shared .align %d .b8 %s[%d];' % (max(t.align, 4), sym, max(t.size, 1)))
            self.bind(d.name, LMem(t, 'shared', sym, 0))
            return
        if 'static' in d.storage and not ('const' in d.storage or 'constexpr' in d.storage):
            self.err('static local variables are not supported')
        init = d.init
        if t is None:
            v = self.rv(init)
            t = v.t
            if isinstance(v, GroupV):
                self.bind(d.name, v)
                return
            d.t = t
            if isinstance(v, V) and isinstance(t, TPtr):
                pass
            st = self.new_var(t, d, space=self.fr.info.prov.get(d.id) if self.fr.info else None)
            if isinstance(st, LReg) and isinstance(t, TPtr):
                st.space = (self.fr.info.prov.get(d.id) if self.fr.info else None) or v.space or 'generic'
            self.store(st, v)
            self.bind(d.name, st)
            return
        if isinstance(t, TRef):
            lv = self.lv(init)
            self.bind(d.name, lv)
            return
        if isinstance(t, TGroup):
            v = self.rv(init) if init is not None else self.err('uninitialised group')
            self.bind(d.name, v)
            return
        if isinstance(t, TFunctor):
            self.bind(d.name, GroupV(t))
            return
        # compile-time constants
        if ('const' in d.storage or 'constexpr' in d.storage) and t.is_arith() and init is not None and \
                not isinstance(init, EInit):
            r = const_value(init, self.env, self.const_env())
            if r is not None:
                self.bind(d.name, V(t, imm=_wrap(r[0], t)))
                return
        prov = self.fr.info.prov.get(d.id) if self.fr.info else None
        st = self.new_var(t, d, space=prov)
        if isinstance(st, LReg) and isinstance(t, TPtr):
            st.space = prov or 'generic'
        if init is not None:
            if isinstance(init, EInit):
                if isinstance(t, (TStruct, TArr)):
                    if isinstance(st, LMem) and not small_agg(t):
                        self.init_mem(st, t, init)
                    else:
                        self.store(st, self.agg_init_value(t, init))
                else:
                    self.store(st, self.convert(self.rv(init.items[0]) if init.items else V(INT, imm=0), t))
            elif isinstance(init, EStr) and isinstance(t, TArr):
                data = list(init.v) + [0] * (t.n - len(init.v))
                leaves = dict(((i,), V(t.elem, imm=b)) for i, b in enumerate(data[:t.n]))
                self.store(st, A(t, leaves))
            else:
                v = self.rv(init)
                if isinstance(t, (TStruct, TArr)):
                    if not isinstance(v, A):
                        self.err('bad initialiser for %r' % (t,))
                    self.store(st, v)
                else:
                    v = self.convert(v, t)
                    if isinstance(st, LReg) and isinstance(t, TPtr):
                        st.space = prov or v.space or 'generic'
                    self.store(st, v)
        self.bind(d.name, st)

    def init_mem(self, st, t, init):
        """Large aggregate in memory: zero it, then store the given elements."""
        zero = V(UINT, imm=0)
        off = 0
        while off + 4 <= t.size:
            self.store_scalar(st, UINT, off, zero)
            off += 4
        while off < t.size:
            self.store_scalar(st, UCHAR, off, zero)
            off += 1

        def put(t, base_off, e):
            if isinstance(t, (TArr, TStruct)):
                items = e.items if isinstance(e, EInit) else [e]
                if isinstance(t, TArr):
                    for i, it in enumerate(items):
                        put(t.elem, base_off + i * t.elem.size, it)
                else:
                    for (fname, ft, fo), it in zip(t.fields, items):
                        put(ft, base_off + fo, it)
                return
            if isinstance(e, EInit):
                e = e.items[0] if e.items else None
                if e is None:
                    return
            v = self.rv(e)
            if not (v.is_imm and v.imm == 0):
                self.store_scalar(st, t, base_off, v)
        put(t, 0, init)

    def const_env(self):
        env = {'__ns__': self.ns}
        for scope in self.fr.scopes:
            for k, b in scope.items():
                if isinstance(b, V) and b.is_imm and b.t.is_arith():
                    env[k] = (b.imm, b.t)
        locals_ = set()
        for scope in self.fr.scopes:
            for k, b in scope.items():
                if not (isinstance(b, V) and b.is_imm):
                    locals_.add(k)
        env['__locals__'] = locals_
        return env

    def cond_branch(self, c, false_label):
        """Branch to false_label when condition c (AST) is false."""
        if isinstance(c, EBin) and c.op == '&&':
            self.cond_branch(c.a, false_label)
            self.cond_branch(c.b, false_label)
            return
        if isinstance(c, EBin) and c.op == '||':
            ok = self.fn.label()
            self.true_branch(c.a, ok)
            self.cond_branch(c.b, false_label)
            self.fn.place(ok)
            return
        if isinstance(c, EUnary) and c.op == '!':
            self.true_branch(c.e, false_label)
            return
        v = self.rv(c)
        if isinstance(v, V) and v.is_imm:
            if not v.imm:
                self.emit('bra.uni %s;' % false_label)
            return
        p = self.to_pred(v)
        self.emit('@!%s bra %s;' % (p, false_label))

    def true_branch(self, c, true_label):
        if isinstance(c, EBin) and c.op == '||':
            self.true_branch(c.a, true_label)
            self.true_branch(c.b, true_label)
            return
        if isinstance(c, EBin) and c.op == '&&':
            no = self.fn.label()
            self.cond_branch(c.a, no)
            self.true_branch(c.b, true_label)
            self.fn.place(no)
            return
        if isinstance(c, EUnary) and c.op == '!':
            self.cond_branch(c.e, true_label)
            return
        v = self.rv(c)
        if isinstance(v, V) and v.is_imm:
            if v.imm:
                self.emit('bra.uni %s;' % true_label)
            return
        p = self.to_pred(v)
        self.emit('@%s bra %s;' % (p, true_label))

    def st_SIf(self, s):
        cv = const_value(s.c, self.env, self.const_env())
        if cv is not None:
            self.stmt(s.t if cv[0] else s.e) if (s.t if cv[0] else s.e) is not None else None
            return
        lab_else = self.fn.label()
        self.cond_branch(s.c, lab_else)
        self.fr.scopes.append({})
        self.stmt(s.t)
        self.fr.scopes.pop()
        if s.e is not None:
            lab_end = self.fn.label()
            self.emit('bra.uni %s;' % lab_end)
            self.fn.place(lab_else)
            self.fr.scopes.append({})
            self.stmt(s.e)
            self.fr.scopes.pop()
            self.fn.place(lab_end)
        else:
            self.fn.place(lab_else)

    def st_SWhile(self, s):
        top, end = self.fn.label(), self.fn.label()
        self.fn.place(top)
        self.cond_branch(s.c, end)
        self.fr.breaks.append(end)
        self.fr.conts.append(top)
        self.fr.scopes.append({})
        self.stmt(s.body)
        self.fr.scopes.pop()
        self.fr.breaks.pop()
        self.fr.conts.pop()
        self.emit('bra.uni %s;' % top)
        self.fn.place(end)

    def st_SDo(self, s):
        top, cont, end = self.fn.label(), self.fn.label(), self.fn.label()
        self.fn.place(top)
        self.fr.breaks.append(end)
        self.fr.conts.append(cont)
        self.fr.scopes.append({})
        self.stmt(s.body)
        self.fr.scopes.pop()
        self.fr.breaks.pop()
        self.fr.conts.pop()
        self.fn.place(cont)
        self.true_branch(s.c, top)
        self.fn.place(end)

    def unroll_info(self, s, cenv, written_fn):
        """For '#pragma unroll' loops with constant bounds: (variable name, list of values) or None."""
        if s.unroll is None or s.unroll == 1:
            return None
        init = s.init
        var = None
        start = None
        if isinstance(init, SDecl) and len(init.decls) == 1:
            d = init.decls[0]
            if d.t is not None and d.t.is_int() and d.init is not None and not isinstance(d.init, EInit):
                var = d.name
                r = const_value(d.init, self.env, cenv)
                start = r[0] if r else None
        elif isinstance(init, SExpr) and isinstance(init.e, EAssign) and init.e.op == '=' and \
                isinstance(init.e.l, EName):
            var = init.e.l.name
            r = const_value(init.e.r, self.env, cenv)
            start = r[0] if r else None
            return None     # loop variable declared outside: keep the loop
        if var is None or start is None:
            return None
        c = s.c
        if not isinstance(c, EBin) or c.op not in ('<', '<=', '>', '>=', '!=') or not isinstance(c.a, EName) or \
                c.a.name != var:
            return None
        r = const_value(c.b, self.env, cenv)
        if r is None:
            return None
        bound = r[0]
        st = s.step
        step = None
        if isinstance(st, EUnary) and st.op in ('++pre', '++post', '--pre', '--post') and \
                isinstance(st.e, EName) and st.e.name == var:
            step = 1 if st.op[0] == '+' else -1
        elif isinstance(st, EAssign) and isinstance(st.l, EName) and st.l.name == var and st.op in ('+=', '-=',
                                                                                                    '*=', '<<=',
                                                                                                    '>>=', '/='):
            r = const_value(st.r, self.env, cenv)
            if r is None:
                return None
            k = r[0]
            step = (st.op, k)
        elif isinstance(st, EAssign) and isinstance(st.l, EName) and st.l.name == var and st.op == '=' and \
                isinstance(st.r, EBin) and isinstance(st.r.a, EName) and st.r.a.name == var:
            r = const_value(st.r.b, self.env, cenv)
            if r is None:
                return None
            step = (st.r.op + '=', r[0])
        if step is None or written_fn(var, s.body):
            return None
        vals = []
        v = start
        ops = {'<': lambda a, b: a < b, '<=': lambda a, b: a <= b, '>': lambda a, b: a > b,
               '>=': lambda a, b: a >= b, '!=': lambda a, b: a != b}
        while ops[c.op](v, bound):
            vals.append(v)
            if len(vals) > MAX_UNROLL:
                return None
            if isinstance(step, int):
                v += step
            else:
                op, k = step
                if op == '+=':
                    v += k
                elif op == '-=':
                    v -= k
                elif op == '*=':
                    v *= k
                elif op == '<<=':
                    v <<= k
                elif op == '>>=':
                    v >>= k
                elif op == '/=' and k:
                    v = int(v / k)
                else:
                    return None
                if op in ('*=', '<<=', '>>=', '/=') and v == vals[-1]:
                    return None
        if s.unroll and s.unroll < len(vals):
            return None      # partial unrolling: keep the loop
        return var, vals

    def st_SFor(self, s):
        self.fr.scopes.append({})
        try:
            info = self.unroll_info(s, self.const_env(), name_written)
            if info is not None:
                var, vals = info
                d = s.init.decls[0]
                end = self.fn.label()
                self.fr.breaks.append(end)
                for v in vals:
                    nxt = self.fn.label()
                    self.fr.scopes.append({var: V(d.t, imm=_wrap(v, d.t))})
                    self.fr.conts.append(nxt)
                    self.stmt(s.body)
                    self.fr.conts.pop()
                    self.fr.scopes.pop()
                    self.fn.place(nxt)
                self.fr.breaks.pop()
                self.fn.place(end)
                return
            if s.init is not None:
                self.stmt(s.init)
            top, cont, end = self.fn.label(), self.fn.label(), self.fn.label()
            self.fn.place(top)
            if s.c is not None:
                self.cond_branch(s.c, end)
            self.fr.breaks.append(end)
            self.fr.conts.append(cont)
            self.fr.scopes.append({})
            self.stmt(s.body)
            self.fr.scopes.pop()
            self.fr.breaks.pop()
            self.fr.conts.pop()
            self.fn.place(cont)
            if s.step is not None:
                self.rv(s.step)
            self.emit('bra.uni %s;' % top)
            self.fn.place(end)
        finally:
            self.fr.scopes.pop()

    def st_SSwitch(self, s):
        v = self.rv_scalar(s.e)
        t = promote(v.t)
        v = self.in_reg(self.convert(v, t))
        end = self.fn.label()
        cases = []
        default = [None]

        def collect(n):
            if isinstance(n, SCase):
                lab = self.fn.label()
                cases.append((n, lab))
            elif isinstance(n, SDefault):
                default[0] = (n, self.fn.label())
            elif isinstance(n, SBlock):
                for x in n.stmts:
                    collect(x)
            elif isinstance(n, SIf):
                collect(n.t)
                collect(n.e)
            elif isinstance(n, (SWhile, SDo, SFor)):
                collect(n.body)
        collect(s.body)
        labels = {}
        for n, lab in cases:
            cv = const_value(n.v, self.env, self.const_env())
            if cv is None:
                self.err('non-constant case label')
            p = self.preg()
            self.emit('setp.eq.%s %s, %s, %s;' % (op_type(t), p, v.r, fmt_imm(_wrap(cv[0], t), t)))
            self.emit('@%s bra %s;' % (p, lab))
            labels[id(n)] = lab
        if default[0]:
            labels[id(default[0][0])] = default[0][1]
            self.emit('bra.uni %s;' % default[0][1])
        else:
            self.emit('bra.uni %s;' % end)
        self.switch_labels = getattr(self, 'switch_labels', [])
        self.switch_labels.append(labels)
        self.fr.breaks.append(end)
        self.stmt(s.body)
        self.fr.breaks.pop()
        self.switch_labels.pop()
        self.fn.place(end)

    def st_SCase(self, s):
        self.fn.place(self.switch_labels[-1][id(s)])

    def st_SDefault(self, s):
        self.fn.place(self.switch_labels[-1][id(s)])

    def st_SBreak(self, s):
        if not self.fr.breaks:
            self.err('break outside a loop')
        self.emit('bra.uni %s;' % self.fr.breaks[-1])

    def st_SContinue(self, s):
        if not self.fr.conts:
            self.err('continue outside a loop')
        self.emit('bra.uni %s;' % self.fr.conts[-1])

    def st_SReturn(self, s):
        fr = self.fr
        if s.e is not None:
            v = self.rv(s.e)
            if fr.ret_store is not None:
                if isinstance(fr.ret_t, (TStruct, TArr)):
                    if isinstance(s.e, EInit):
                        v = self.agg_init_value(fr.ret_t, s.e)
                    self.store(fr.ret_store, v)
                else:
                    v = self.convert(v, fr.ret_t)
                    if isinstance(v, V) and isinstance(fr.ret_t, TPtr):
                        fr.ret_space = join(getattr(fr, 'ret_space', None), v.space or 'generic')
                    self.store(fr.ret_store, v)
        self.emit('bra.uni %s;' % fr.ret_label)

    # ---- kernels ------------------------------------------------------------------------
    def kernel(self, fd):
        self.fn = Fn()
        kname = 'pk_k_' + ident(fd.name)
        params = {}
        binds = []
        decls = []
        for i, p in enumerate(fd.params):
            pt = unqual(p.t)
            pname = '%s_p%d' % (kname, i)
            if isinstance(pt, TStruct):
                decls.append('.param .align %d .b8 %s[%d]' % (max(pt.align, 4), pname, pt.size))
            elif isinstance(pt, (TPtr, TInt, TFloat)):
                mt = mem_type(pt)
                if mt == 'u8' and pt is BOOL:
                    mt = 'u8'
                decls.append('.param .%s %s' % (mt, pname))
            else:
                self.err('unsupported kernel parameter type %r' % (pt,))
            prov = 'global' if isinstance(pt, TPtr) else None
            binds.append((p, pt, pname, prov))
            params[p.name] = (p, prov, False)
        psize = 0
        for _, pt, _, _ in binds:
            psize = (psize + max(pt.align, 1) - 1) // max(pt.align, 1) * max(pt.align, 1) + pt.size
        if psize > 4352:    # large kernel parameters (up to 32764 bytes): Volta and newer, PTX ISA 8.1
            if psize > 32764:
                self.err('kernel parameters take %d bytes (the limit is 32764)' % psize)
            self.m.need_sm = max(self.m.need_sm, 70)
            self.m.need_ptx = max(self.m.need_ptx, (8, 1))
        info = Analysis(self, params)
        self.ns = list(fd.ns or [])
        info.run(fd.body)
        frame = Frame(fd.name, info, self.ns)
        self.frames = [frame]
        frame.ret_label = self.fn.label()
        frame.ret_t = VOID
        for p, pt, pname, prov in binds:
            if p.name is None:
                continue
            if isinstance(pt, TStruct):
                src = LMem(pt, 'param', pname, 0, True)
                if small_agg(pt) and p.id not in info.mem:
                    v = self.load(src)
                    if p.id in info.written:
                        st = self.new_var(pt, p)
                        self.store(st, v)
                        self.bind(p.name, st)
                    else:
                        self.bind(p.name, v)
                elif p.id in info.mem or p.id in info.written:
                    st = LMem(pt, 'local', '__pk_depot', self.fn.slot(pt.size, pt.align))
                    self.copy_mem(st, src, pt.size, pt.align)
                    self.bind(p.name, st)
                else:
                    self.bind(p.name, src)
                continue
            r = self.reg(pt)
            self.emit('ld.param.%s %s, [%s];' % (mem_type(pt), r, pname))
            v = V(pt, r, space=prov)
            if p.id in info.written or p.id in info.mem:
                st = self.new_var(pt, p, space=info.prov.get(p.id, prov))
                if isinstance(st, LReg) and isinstance(pt, TPtr):
                    st.space = info.prov.get(p.id, prov) or 'generic'
                self.store(st, v)
                self.bind(p.name, st)
            else:
                self.bind(p.name, v)
        self.block_body(fd.body)
        self.fn.place(frame.ret_label)
        self.emit('ret;')
        directives = []
        for item in fd.storage:
            if isinstance(item, tuple) and item[0] == 'launch_bounds':
                vals = [x for x in item[1] if x != ',']
                try:
                    nums = [int(self.env.const_eval(Parser([('num', x)] if re.match(r'^\d+$', x) else
                                                            [('id', x)], self.env).expr())) for x in vals[:2]]
                    if nums:
                        directives.append('.maxntid %d, 1, 1' % nums[0])
                    if len(nums) > 1 and nums[1] > 0:
                        directives.append('.minnctapersm %d' % nums[1])
                except (ParseError, CompileError, ValueError):
                    pass
        regs = []
        names = {'p': '.pred', 'r': '.b32', 'rd': '.b64', 'f': '.f32', 'fd': '.f64'}
        for cls, n in self.fn.n.items():
            if n > 1:
                regs.append('    .reg %s %%%s<%d>;' % (names[cls], cls, n))
        if self.fn.depot:
            regs.append('    .local .align %d .b8 __pk_depot[%d];' % (self.fn.depot_align, self.fn.depot))
        head = '.visible .entry %s(\n    %s\n)%s' % (kname, ',\n    '.join(decls),
                                                  ('\n' + '\n'.join(directives)) if directives else '')
        if not decls:
            head = '.visible .entry %s()%s' % (kname, ('\n' + '\n'.join(directives)) if directives else '')
        return head + '\n{\n' + '\n'.join(regs) + '\n' + '\n'.join(self.fn.lines) + '\n}\n'

    def copy_mem(self, dst, src, size, align):
        off = 0
        unit, t = (8, ULONG) if align >= 8 else ((4, UINT) if align >= 4 else (1, UCHAR))
        while off + unit <= size:
            v = self.load_scalar(src, t, off)
            self.store_scalar(dst, t, off, v)
            off += unit
        while off < size:
            v = self.load_scalar(src, UCHAR, off)
            self.store_scalar(dst, UCHAR, off, v)
            off += 1


class NeedMemory(CompileError):
    def __init__(self, lv):
        CompileError.__init__(self, 'dynamic index into a register array')
        self.lv = lv


class NotLvalue(CompileError):
    pass


def mov_type(t):
    if t.is_float():
        return 'f32' if t.size == 4 else 'f64'
    if isinstance(t, TPtr) or t.size == 8:
        return 'u64'
    return 'u32'


from pk_ptx_builtins import BUILTINS, group_method   # noqa: E402  (needs the classes above)


# ---------------------------------------------------------------------------
# Driver: compile a translation unit
# ---------------------------------------------------------------------------
_PRELUDE = None


def load_prelude(env):
    """Parse ptx_prelude.cu into FuncDefs (by name) using the TU's environment."""
    path = os.path.join(HERE, 'ptx_prelude.cu')
    with open(path, encoding='utf-8') as f:
        src = f.read()
    src = re.sub(r'//[^\n]*', '', src)            # not preprocessed: drop comments here
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    from pk_ocl import split_items
    toks = [t for t in tokenize(src) if t[0] not in ('ws', 'nl')]
    out = {}
    for it in split_items(toks):
        try:
            p = Parser(it.toks, env, it.ns)
            for d in p.external():
                if isinstance(d, FuncDef):
                    out.setdefault(d.name, []).append(d)
        except ParseError as e:
            raise CompileError('prelude: %s' % e)
    return out


def find_ptxas():
    p = os.environ.get('PK_PTXAS')
    if p == 'none':
        return None
    for c in (p, 'ptxas', '/usr/local/cuda/bin/ptxas'):
        if c and shutil.which(c):
            return shutil.which(c)
    return None


def ptxas_check(text, target, ptxas):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'm.ptx')
        with open(src, 'w', encoding='utf-8') as f:
            f.write(text)
        r = subprocess.run([ptxas, '-arch=sm_%d' % target, src, '-o', os.path.join(d, 'm.cubin')],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', errors='replace')
        return r.returncode == 0, r.stderr.strip()


def generate_ptx(dev_src, target=60, log=None, check=True):
    """Compile device code to PTX. Returns dict(ptx_ok, ptx_src, kernels, reason, skipped)."""
    from pk_ocl import split_items, classify, Ctx as OclCtx
    result = {'ptx_ok': False, 'ptx_src': None, 'kernels': [], 'reason': None, 'skipped': [], 'target': target}
    toks = [t for t in tokenize(dev_src) if t[0] not in ('ws', 'nl')]
    items = split_items(toks)
    octx = OclCtx()
    env = Env()
    kernels = []
    for it in items:
        try:
            classify(it, octx)
        except Exception:
            it.kind = 'other'
    for it in items:
        if it.kind in ('hostfunc', 'other', 'template', 'shared_global_x'):
            if it.kind == 'template' and any(t[1] == '__global__' for t in it.toks):
                result['skipped'].append(('template kernel', 'templates are not supported yet'))
            continue
        try:
            p = Parser(it.toks, env, it.ns)
            decls = p.external()
        except (ParseError, IndexError, KeyError, AttributeError, TypeError) as e:
            if it.kind in ('kernel', 'devfunc', 'kernel_decl', 'devfunc_decl'):
                result['skipped'].append((it.name or '?', 'parse: %s' % e))
                if log:
                    log('PTX: cannot parse %s: %s' % (it.name, e))
            continue
        for d in decls:
            if isinstance(d, FuncDef):
                env.funcs.setdefault(d.name, []).append(d)
                if d.ns:
                    env.funcs.setdefault('::'.join(list(d.ns) + [d.name]), []).append(d)
                if d.kind == 'kernel' and d.body is not None:
                    kernels.append(d)
            elif isinstance(d, VarDecl):
                st = d.storage
                if '__device__' in st or '__constant__' in st or '__shared__' in st or '__managed__' in st:
                    env.globals[d.name] = d
                    if d.ns:
                        env.globals['::'.join(list(d.ns) + [d.name])] = d
                elif 'const' in st or 'constexpr' in st:
                    env.consts[d.name] = d
                    if d.ns:
                        env.consts['::'.join(list(d.ns) + [d.name])] = d
    managed = sorted(set(g.name for g in env.globals.values() if '__managed__' in g.storage))
    if managed:   # the host would need to see the device copy: not supported yet
        result['reason'] = "__managed__ variable '%s' is not supported on the PTX back end yet" % managed[0]
        return result
    try:
        prelude = load_prelude(env)
    except CompileError as e:
        result['reason'] = str(e)
        return result
    module = Module(env, target)
    ok_kernels = []
    for fd in kernels:
        saved_decls = (list(module.decls), set(module.declared), dict(module.strings), module.dynsmem)
        try:
            text = compile_kernel(module, env, prelude, fd)
            module.entries.append(text)
            ok_kernels.append(fd.name)
        except (CompileError, ParseError, RecursionError) as e:
            module.decls, module.declared, module.strings, module.dynsmem = saved_decls
            result['skipped'].append((fd.name, str(e)))
            if log:
                log('PTX: kernel %s stays off the PTX path: %s' % (fd.name, e))
    if module.dynsmem:
        module.declare('pk_dynsmem', '.extern .shared .align 16 .b8 pk_dynsmem[];')
    for g in set(env.globals.values()):   # every device variable gets storage (the host may copy to it)
        if '__shared__' in g.storage:
            continue
        sym = 'pk_g_' + ident(g.name)
        space = 'const' if '__constant__' in g.storage else 'global'
        try:
            module.declare(sym, '.%s .align %d .b8 %s[%d];' % (space, max(g.t.align, 4), sym, max(g.t.size, 1)))
        except AttributeError:
            pass
    text = module.text()
    result['kernels'] = ok_kernels
    seen = set()
    result['symbols'] = []
    for g in env.globals.values():      # (host reference, name) of every __device__/__constant__ variable
        if '__shared__' in g.storage or id(g) in seen or '__managed__' in g.storage:
            continue
        seen.add(id(g))
        ns = [x for x in (getattr(g, 'ns', None) or []) if x != '(anon)']
        result['symbols'].append(('::' + '::'.join(ns + [g.name]), g.name))
    if not kernels:      # nothing to compile; a module is still needed for the file's device variables
        result['ptx_ok'] = True
        result['reason'] = 'no kernels'
        result['ptx_src'] = text if result['symbols'] else None
        return result
    if not ok_kernels:
        result['reason'] = 'no kernel could be compiled to PTX'
        result['ptx_src'] = text
        return result
    ptxas = find_ptxas() if check else None
    if ptxas:
        ok, msg = ptxas_check(text, max(target, module.need_sm), ptxas)
        if not ok:
            result['reason'] = 'ptxas: ' + msg[:400]
            result['ptx_src'] = text
            if log:
                log('PTX: ptxas rejected the module: %s' % msg[:2000])
            return result
    result['ptx_ok'] = True
    result['ptx_src'] = text
    result['target'] = max(target, module.need_sm)
    return result


def compile_kernel(module, env, prelude, fd):
    forced = set()
    for _ in range(8):
        g = Gen(module, env, prelude)
        g.forced_mem = forced
        orig_run = Analysis.run

        def run(self, body, iterations=4, _f=forced):
            orig_run(self, body, iterations)
            self.mem |= _f
        Analysis.run = run
        try:
            return g.kernel(fd)
        except NeedMemory as e:
            ids = find_decl_ids_for(e.lv, g)
            if not ids or ids <= forced:
                raise CompileError('dynamic index into a register array')
            forced |= ids
        finally:
            Analysis.run = orig_run
    raise CompileError('could not place local arrays')


def find_decl_ids_for(lv, g):
    """Find which local declarations own the register store of lv (to force them into memory)."""
    ids = set()
    for fr in g.frames:
        for scope in fr.scopes:
            for name, b in scope.items():
                if isinstance(b, LReg) and b.store is lv.store:
                    for d_id in getattr(fr, 'decl_ids', {}).get(name, ()):
                        ids.add(d_id)
    if not ids:
        ids = set(getattr(lv, 'decl_ids', ()) or ())
    return ids or set(LREG_OWNERS.get(id(lv.store), ()))


LREG_OWNERS = {}
_orig_new_var = Gen.new_var


def _new_var_tracked(self, t, decl=None, space=None):
    st = _orig_new_var(self, t, decl, space)
    if decl is not None and isinstance(st, LReg):
        LREG_OWNERS[id(st.store)] = {decl.id}
    return st


Gen.new_var = _new_var_tracked
