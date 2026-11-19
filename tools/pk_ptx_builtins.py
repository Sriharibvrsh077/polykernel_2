"""CUDA device built-ins for pk_ptx: each maps to PTX instructions (per the public PTX ISA)."""
import re

from pk_cparse import (TPtr, TArr, TStruct, TGroup, TFunctor, TInt, TFloat, TVoid, VOID, BOOL, CHAR, INT, UINT, LONG,
                       ULONG, LLONG, ULLONG, FLOAT, DOUBLE, VEC, EUnary, EName, ENum, unqual)
import pk_ptx as P

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
LOG2_10 = 3.321928094887362
LOG10_2 = 0.30102999566398120


def V(*a, **kw):
    return P.V(*a, **kw)


def sv(g, a):
    return g.rv_scalar(a)


def cv(g, a, t):
    return g.convert(sv(g, a), t)


def reg_in(g, a, t):
    return g.in_reg(cv(g, a, t))


def ftype(g, a, default=DOUBLE):
    """Float type for a generic math function argument (float stays float, everything else double)."""
    t = g.peek_type(a)
    if t is FLOAT:
        return FLOAT
    if t is DOUBLE:
        return DOUBLE
    return default


def ret_void():
    return V(VOID, imm=0)


def instr1(g, name, t, a):
    x = reg_in(g, a, t)
    r = g.reg(t)
    g.emit('%s %s, %s;' % (name, r, x.r))
    return V(t, r)


# ---- synchronisation ---------------------------------------------------------------
def b_syncthreads(g, args, targs):
    g.emit('bar.sync 0;')
    return ret_void()


def b_syncthreads_red(kind):
    def h(g, args, targs):
        p = g.to_pred(sv(g, args[0]))
        if kind == 'popc':
            r = g.fn.reg('r')
            g.emit('bar.red.popc.u32 %s, 0, %s;' % (r, p))
            return V(INT, r)
        q = g.preg()
        g.emit('bar.red.%s.pred %s, 0, %s;' % (kind, q, p))
        return g.in_reg(V(BOOL, q)) if False else g.convert(V(BOOL, q), INT)
    return h


def b_syncwarp(g, args, targs):
    m = cv(g, args[0], UINT) if args else V(UINT, imm=0xffffffff)
    g.emit('bar.warp.sync %s;' % g.operand(m))
    return ret_void()


def b_fence(scope):
    def h(g, args, targs):
        g.emit('membar.%s;' % scope)
        return ret_void()
    return h


def b_trap(g, args, targs):
    g.emit('trap;')
    return ret_void()


# ---- atomics ---------------------------------------------------------------------------
def atomic(op, scope=''):
    def h(g, args, targs):
        pv = sv(g, args[0])
        if not isinstance(pv.t, TPtr):
            g.err('atomic on a non-pointer')
        t = unqual(pv.t.to)
        if isinstance(t, TVoid) or not t.is_arith():
            g.err('atomic on %r' % (t,))
        lv = g.deref(pv, t)
        sp = '' if lv.space == 'generic' else '.' + lv.space
        sc = {'block': '.cta', 'system': '.sys', '': ''}[scope]
        bits = t.size * 8
        if t.size not in (4, 8):
            g.err('atomic on a %d-byte type' % t.size)
        if t.is_float():
            ty = 'f%d' % bits
        else:
            ty = None
        if op == 'cas':
            cmpv = g.in_reg(cv(g, args[1], t))
            val = g.in_reg(cv(g, args[2], t))
            bt = 'b%d' % bits
            r = g.reg(t)
            if t.is_float():
                ci, vi, ri = g.fn.reg('r' if bits == 32 else 'rd'), g.fn.reg('r' if bits == 32 else 'rd'), \
                    g.fn.reg('r' if bits == 32 else 'rd')
                g.emit('mov.%s %s, %s;' % (bt, ci, cmpv.r))
                g.emit('mov.%s %s, %s;' % (bt, vi, val.r))
                g.emit('atom%s%s.cas.%s %s, %s, %s, %s;' % (sc, sp, bt, ri, g.addr_text(lv), ci, vi))
                g.emit('mov.%s %s, %s;' % (bt, r, ri))
            else:
                g.emit('atom%s%s.cas.%s %s, %s, %s, %s;' % (sc, sp, bt, r, g.addr_text(lv), cmpv.r, val.r))
            return V(t, r)
        val = cv(g, args[1], t) if op not in ('inc', 'dec') else cv(g, args[1], UINT)
        o = op
        if op == 'sub':
            if t.is_float():
                g.err('atomicSub on floating point')
            val = g.arith('-', V(t, imm=0), val) if not val.is_imm else V(t, imm=P._wrap(-val.imm, t))
            o = 'add'
        if o == 'add':
            ty = ty or ('u%d' % bits)
        elif o in ('min', 'max'):
            if t.is_float():
                g.err('atomicMin/Max on floating point')
            ty = ('s' if t.signed else 'u') + str(bits)
        elif o in ('and', 'or', 'xor', 'exch'):
            ty = 'b%d' % bits
        elif o in ('inc', 'dec'):
            if t.size != 4 or t.signed:
                g.err('atomicInc/Dec need unsigned int')
            ty = 'u32'
        r = g.reg(t)
        if o == 'exch' and t.is_float():
            vi = g.fn.reg('r' if bits == 32 else 'rd')
            g.emit('mov.b%d %s, %s;' % (bits, vi, g.in_reg(val).r))
            ri = g.fn.reg('r' if bits == 32 else 'rd')
            g.emit('atom%s%s.exch.b%d %s, %s, %s;' % (sc, sp, bits, ri, g.addr_text(lv), vi))
            g.emit('mov.b%d %s, %s;' % (bits, r, ri))
            return V(t, r)
        g.emit('atom%s%s.%s.%s %s, %s, %s;' % (sc, sp, o, ty, r, g.addr_text(lv), g.operand(val)))
        return V(t, r)
    return h


# ---- warp intrinsics ------------------------------------------------------------------
def shfl_c(g, width_arg, mode):
    if width_arg is None:
        w = V(INT, imm=32)
    else:
        w = cv(g, width_arg, INT)
    if w.is_imm:
        c = ((32 - int(w.imm)) << 8) | (0 if mode == 'up' else 0x1f)
        return str(c)
    t = g.fn.reg('r')
    g.emit('sub.s32 %s, 32, %s;' % (t, w.r))
    g.emit('shl.b32 %s, %s, 8;' % (t, t))
    if mode != 'up':
        g.emit('or.b32 %s, %s, 31;' % (t, t))
    return t


def do_shfl(g, mode, mask, val, b, c):
    """Shuffle a scalar value of any width; returns V of val.t."""
    t = val.t
    if t is BOOL or (t.is_int() and t.size < 4):
        val = g.convert(val, INT)
        t = INT
    val = g.in_reg(val)
    m = g.operand(mask)
    bb = g.operand(b) if not isinstance(b, str) else b
    if t.size == 8:
        lo, hi = g.fn.reg('r'), g.fn.reg('r')
        g.emit('mov.b64 {%s, %s}, %s;' % (lo, hi, val.r))
        lo2, hi2 = g.fn.reg('r'), g.fn.reg('r')
        g.emit('shfl.sync.%s.b32 %s, %s, %s, %s, %s;' % (mode, lo2, lo, bb, c, m))
        g.emit('shfl.sync.%s.b32 %s, %s, %s, %s, %s;' % (mode, hi2, hi, bb, c, m))
        r = g.reg(t)
        g.emit('mov.b64 %s, {%s, %s};' % (r, lo2, hi2))
        return V(t, r)
    if t.is_float():
        x = g.fn.reg('r')
        g.emit('mov.b32 %s, %s;' % (x, val.r))
        y = g.fn.reg('r')
        g.emit('shfl.sync.%s.b32 %s, %s, %s, %s, %s;' % (mode, y, x, bb, c, m))
        r = g.reg(t)
        g.emit('mov.b32 %s, %s;' % (r, y))
        return V(t, r)
    r = g.fn.reg('r')
    g.emit('shfl.sync.%s.b32 %s, %s, %s, %s, %s;' % (mode, r, val.r, bb, c, m))
    return V(t, r)


def shfl(mode, legacy=False):
    def h(g, args, targs):
        if legacy:
            mask = V(UINT, imm=0xffffffff)
            rest = args
        else:
            mask = cv(g, args[0], UINT)
            rest = args[1:]
        val = sv(g, rest[0])
        b = cv(g, rest[1], UINT)
        c = shfl_c(g, rest[2] if len(rest) > 2 else None, mode)
        return do_shfl(g, mode, mask, val, b, c)
    return h


def vote(kind, legacy=False):
    def h(g, args, targs):
        if legacy:
            mask, pa = V(UINT, imm=0xffffffff), args[0]
        else:
            mask, pa = cv(g, args[0], UINT), args[1]
        p = g.to_pred(sv(g, pa))
        if kind == 'ballot':
            r = g.fn.reg('r')
            g.emit('vote.sync.ballot.b32 %s, %s, %s;' % (r, p, g.operand(mask)))
            return V(UINT, r)
        q = g.preg()
        g.emit('vote.sync.%s.pred %s, %s, %s;' % (kind, q, p, g.operand(mask)))
        return g.convert(V(BOOL, q), INT)
    return h


def b_activemask(g, args, targs):
    r = g.fn.reg('r')
    g.emit('activemask.b32 %s;' % r)
    return V(UINT, r)


def b_lanemask(which):
    def h(g, args, targs):
        r = g.fn.reg('r')
        g.emit('mov.u32 %s, %%lanemask_%s;' % (r, which))
        return V(UINT, r)
    return h


def b_laneid(g, args, targs):
    r = g.fn.reg('r')
    g.emit('mov.u32 %s, %%laneid;' % r)
    return V(UINT, r)


def match_operand(g, v):
    """An integer register holding the bits of v (floats are compared bit for bit, as match.sync does)."""
    v = g.in_reg(v)
    bits = 64 if v.t.size == 8 else 32
    if v.t.is_float():
        x = g.fn.reg('r' if bits == 32 else 'rd')
        g.emit('mov.b%d %s, %s;' % (bits, x, v.r))
        return V(UINT if bits == 32 else ULLONG, x)
    if v.t is BOOL or v.t.size < 4:
        return g.in_reg(g.convert(v, INT))
    return v


def b_match_any(g, args, targs):
    mask = cv(g, args[0], UINT)
    return match_any_mask(g, mask, args[1])


def b_match_all(g, args, targs):
    mask = g.in_reg(cv(g, args[0], UINT))
    v = match_operand(g, sv(g, args[1]))
    bits = 64 if v.t.size == 8 else 32
    r = g.fn.reg('r')
    p = g.preg()
    if g.m.target >= 70:
        g.emit('match.all.sync.b%d %s|%s, %s, %s;' % (bits, r, p, v.r, mask.r))
    else:   # all members agree when the lanes matching this lane's value are the whole mask
        same = match_any_mask(g, mask, v)
        g.emit('setp.eq.u32 %s, %s, %s;' % (p, same.r, mask.r))
        g.emit('selp.b32 %s, %s, 0, %s;' % (r, mask.r, p))
    if len(args) > 2:
        a = args[2]
        lv = g.lv(a.e) if isinstance(a, EUnary) and a.op == '&' else g.deref(sv(g, a), INT)
        g.store(lv, g.convert(V(BOOL, p), INT))
    return V(UINT, r)


def b_nanosleep(g, args, targs):
    v = cv(g, args[0], UINT)
    if g.m.target >= 70:
        g.emit('nanosleep.u32 %s;' % g.operand(v))
    return ret_void()


# ---- bit manipulation -------------------------------------------------------------------
def b_popc(bits):
    def h(g, args, targs):
        x = reg_in(g, args[0], UINT if bits == 32 else ULLONG)
        r = g.fn.reg('r')
        g.emit('popc.b%d %s, %s;' % (bits, r, x.r))
        return V(INT, r)
    return h


def b_clz(bits):
    def h(g, args, targs):
        x = reg_in(g, args[0], UINT if bits == 32 else ULLONG)
        r = g.fn.reg('r')
        g.emit('clz.b%d %s, %s;' % (bits, r, x.r))
        return V(INT, r)
    return h


def b_ffs(bits):
    def h(g, args, targs):
        x = reg_in(g, args[0], UINT if bits == 32 else ULLONG)
        t = g.fn.reg('r' if bits == 32 else 'rd')
        g.emit('brev.b%d %s, %s;' % (bits, t, x.r))
        n = g.fn.reg('r')
        g.emit('clz.b%d %s, %s;' % (bits, n, t))
        g.emit('add.u32 %s, %s, 1;' % (n, n))
        p = g.preg()
        g.emit('setp.eq.u%d %s, %s, 0;' % (bits, p, x.r))
        r = g.fn.reg('r')
        g.emit('selp.u32 %s, 0, %s, %s;' % (r, n, p))
        return V(INT, r)
    return h


def b_brev(bits):
    def h(g, args, targs):
        t = UINT if bits == 32 else ULLONG
        x = reg_in(g, args[0], t)
        r = g.reg(t)
        g.emit('brev.b%d %s, %s;' % (bits, r, x.r))
        return V(t, r)
    return h


def b_byte_perm(g, args, targs):
    a, b, s = (reg_in(g, x, UINT) for x in args[:3])
    r = g.fn.reg('r')
    g.emit('prmt.b32 %s, %s, %s, %s;' % (r, a.r, b.r, s.r))
    return V(UINT, r)


def b_funnel(direction, mode):
    def h(g, args, targs):
        lo, hi, sh = (reg_in(g, x, UINT) for x in args[:3])
        r = g.fn.reg('r')
        g.emit('shf.%s.%s.b32 %s, %s, %s, %s;' % (direction, mode, r, lo.r, hi.r, sh.r))
        return V(UINT, r)
    return h


def b_mulx(instr, t):
    def h(g, args, targs):
        a = reg_in(g, args[0], t)
        b = cv(g, args[1], t)
        r = g.reg(t)
        g.emit('%s %s, %s, %s;' % (instr, r, a.r, g.operand(b)))
        return V(t, r)
    return h


def b_sad(t):
    def h(g, args, targs):
        a, b = reg_in(g, args[0], t), reg_in(g, args[1], t)
        c = reg_in(g, args[2], UINT)
        r = g.fn.reg('r')
        g.emit('sad.%s %s, %s, %s, %s;' % ('s32' if t.signed else 'u32', r, a.r, b.r, c.r))
        return V(UINT, r)
    return h


def b_hadd(signed, rnd):
    def h(g, args, targs):
        t = INT if signed else UINT
        a = g.convert(cv(g, args[0], t), LLONG if signed else ULLONG)
        b = g.convert(cv(g, args[1], t), LLONG if signed else ULLONG)
        s = g.arith('+', a, b)
        if rnd:
            s = g.arith('+', s, V(s.t, imm=1))
        s = g.in_reg(s)
        r = g.fn.reg('rd')
        g.emit('shr.%s64 %s, %s, 1;' % ('s' if signed else 'u', r, s.r))
        return g.convert(V(s.t, r), t)
    return h


# ---- type punning and conversions ---------------------------------------------------------
def b_bitcast(src, dst):
    def h(g, args, targs):
        x = reg_in(g, args[0], src)
        r = g.reg(dst)
        g.emit('mov.b%d %s, %s;' % (src.size * 8, r, x.r))
        return V(dst, r)
    return h


def b_double2hi(hi):
    def h(g, args, targs):
        x = reg_in(g, args[0], DOUBLE)
        lo_r, hi_r = g.fn.reg('r'), g.fn.reg('r')
        g.emit('mov.b64 {%s, %s}, %s;' % (lo_r, hi_r, x.r))
        return V(INT, hi_r if hi else lo_r)
    return h


def b_hiloint2double(g, args, targs):
    hi = reg_in(g, args[0], INT)
    lo = reg_in(g, args[1], INT)
    r = g.reg(DOUBLE)
    g.emit('mov.b64 %s, {%s, %s};' % (r, lo.r, hi.r))
    return V(DOUBLE, r)


CONV_T = {'float': FLOAT, 'double': DOUBLE, 'int': INT, 'uint': UINT, 'll': LLONG, 'ull': ULLONG}
RND_F2I = {'rn': 'rni', 'rz': 'rzi', 'ru': 'rpi', 'rd': 'rmi'}
RND_F = {'rn': 'rn', 'rz': 'rz', 'ru': 'rp', 'rd': 'rm'}


def conversion(src, dst, rnd):
    st, dt = CONV_T[src], CONV_T[dst]

    def h(g, args, targs):
        x = reg_in(g, args[0], st)
        r = g.reg(dt)
        if st.is_float() and dt.is_int():
            m = RND_F2I[rnd]
        elif dt.is_float() and (st.is_int() or st.size > dt.size):
            m = RND_F[rnd]
        else:
            m = None
        sty = P.op_type(st) if st.is_float() else ('s' if st.signed else 'u') + str(st.size * 8)
        dty = P.op_type(dt) if dt.is_float() else ('s' if dt.signed else 'u') + str(dt.size * 8)
        if m is None:
            g.emit('cvt.%s.%s %s, %s;' % (dty, sty, r, x.r))
        else:
            g.emit('cvt.%s.%s.%s %s, %s;' % (m, dty, sty, r, x.r))
        return V(dt, r)
    return h


# ---- math ----------------------------------------------------------------------------------
def f_unary(instr32, instr64=None, fixed=None):
    """Unary float function; fixed = FLOAT/DOUBLE forces the type, else chosen from the argument."""
    def h(g, args, targs):
        t = fixed or ftype(g, args[0])
        ins = instr32 if t is FLOAT else instr64
        if ins is None:
            g.err('no double version')
        return instr1(g, ins, t, args[0])
    return h


def f_binary(op32, op64=None, fixed=None):
    def h(g, args, targs):
        t = fixed or (FLOAT if ftype(g, args[0]) is FLOAT and ftype(g, args[1]) is FLOAT else DOUBLE)
        a = reg_in(g, args[0], t)
        b = cv(g, args[1], t)
        r = g.reg(t)
        g.emit('%s %s, %s, %s;' % (op32 if t is FLOAT else op64, r, a.r, g.operand(b)))
        return V(t, r)
    return h


def f_fma(rnd='rn', fixed=None):
    def h(g, args, targs):
        t = fixed or (FLOAT if all(ftype(g, a) is FLOAT for a in args[:3]) else DOUBLE)
        a, b, c = (reg_in(g, x, t) for x in args[:3])
        r = g.reg(t)
        g.emit('fma.%s.%s %s, %s, %s, %s;' % (rnd, P.op_type(t), r, a.r, b.r, c.r))
        return V(t, r)
    return h


def scale_ex2(g, x, k):
    """ex2.approx(x * k) in float."""
    t = g.fn.reg('f')
    g.emit('mul.f32 %s, %s, %s;' % (t, x.r, P.fmt_imm(k, FLOAT)))
    r = g.fn.reg('f')
    g.emit('ex2.approx.f32 %s, %s;' % (r, t))
    return V(FLOAT, r)


def lg2_scaled(g, x, k):
    t = g.fn.reg('f')
    g.emit('lg2.approx.f32 %s, %s;' % (t, x.r))
    r = g.fn.reg('f')
    g.emit('mul.f32 %s, %s, %s;' % (r, t, P.fmt_imm(k, FLOAT)))
    return V(FLOAT, r)


def b_fast(kind):
    def h(g, args, targs):
        x = reg_in(g, args[0], FLOAT)
        if kind == 'exp':
            return scale_ex2(g, x, LOG2E)
        if kind == 'exp10':
            return scale_ex2(g, x, LOG2_10)
        if kind == 'log':
            return lg2_scaled(g, x, LN2)
        if kind == 'log10':
            return lg2_scaled(g, x, LOG10_2)
        if kind == 'log2':
            r = g.fn.reg('f')
            g.emit('lg2.approx.f32 %s, %s;' % (r, x.r))
            return V(FLOAT, r)
        if kind in ('sin', 'cos'):
            r = g.fn.reg('f')
            g.emit('%s.approx.f32 %s, %s;' % (kind, r, x.r))
            return V(FLOAT, r)
        if kind == 'tan':
            s, c = g.fn.reg('f'), g.fn.reg('f')
            g.emit('sin.approx.f32 %s, %s;' % (s, x.r))
            g.emit('cos.approx.f32 %s, %s;' % (c, x.r))
            r = g.fn.reg('f')
            g.emit('div.approx.f32 %s, %s, %s;' % (r, s, c))
            return V(FLOAT, r)
        g.err(kind)
    return h


def b_powf_fast(g, args, targs):
    x = reg_in(g, args[0], FLOAT)
    y = reg_in(g, args[1], FLOAT)
    l = g.fn.reg('f')
    g.emit('lg2.approx.f32 %s, %s;' % (l, x.r))
    m = g.fn.reg('f')
    g.emit('mul.f32 %s, %s, %s;' % (m, l, y.r))
    r = g.fn.reg('f')
    g.emit('ex2.approx.f32 %s, %s;' % (r, m))
    return V(FLOAT, r)


def out_store(g, a, v):
    if isinstance(a, EUnary) and a.op == '&':
        lv = g.lv(a.e)
    else:
        pv = sv(g, a)
        lv = g.deref(pv)
    g.store(lv, g.convert(v, unqual(lv.t)))


def b_sincosf_fast(g, args, targs):
    x = reg_in(g, args[0], FLOAT)
    s, c = g.fn.reg('f'), g.fn.reg('f')
    g.emit('sin.approx.f32 %s, %s;' % (s, x.r))
    g.emit('cos.approx.f32 %s, %s;' % (c, x.r))
    out_store(g, args[1], V(FLOAT, s))
    out_store(g, args[2], V(FLOAT, c))
    return ret_void()


def b_saturate(g, args, targs):
    x = reg_in(g, args[0], FLOAT)
    a = g.fn.reg('f')
    g.emit('max.f32 %s, %s, 0f00000000;' % (a, x.r))
    r = g.fn.reg('f')
    g.emit('min.f32 %s, %s, 0f3F800000;' % (r, a))
    return V(FLOAT, r)


def b_fdividef(g, args, targs):
    a = reg_in(g, args[0], FLOAT)
    b = reg_in(g, args[1], FLOAT)
    r = g.fn.reg('f')
    g.emit('div.approx.f32 %s, %s, %s;' % (r, a.r, b.r))
    return V(FLOAT, r)


def b_rounded(op, t, rnd):
    def h(g, args, targs):
        vals = [reg_in(g, a, t) for a in args]
        r = g.reg(t)
        ty = P.op_type(t)
        if op == 'fma':
            g.emit('fma.%s.%s %s, %s, %s, %s;' % (rnd, ty, r, vals[0].r, vals[1].r, vals[2].r))
        elif op in ('rcp', 'sqrt'):
            g.emit('%s.%s.%s %s, %s;' % (op, rnd, ty, r, vals[0].r))
        else:
            g.emit('%s.%s.%s %s, %s, %s;' % (op, rnd, ty, r, vals[0].r, vals[1].r))
        return V(t, r)
    return h


def b_isnan(g, args, targs):
    t = ftype(g, args[0])
    x = reg_in(g, args[0], t)
    p = g.preg()
    g.emit('testp.notanumber.%s %s, %s;' % (P.op_type(t), p, x.r))
    return g.convert(V(BOOL, p), INT)


def b_isinf(g, args, targs):
    t = ftype(g, args[0])
    x = reg_in(g, args[0], t)
    p = g.preg()
    g.emit('testp.infinite.%s %s, %s;' % (P.op_type(t), p, x.r))
    return g.convert(V(BOOL, p), INT)


def b_isfinite(g, args, targs):
    t = ftype(g, args[0])
    x = reg_in(g, args[0], t)
    p = g.preg()
    g.emit('testp.finite.%s %s, %s;' % (P.op_type(t), p, x.r))
    return g.convert(V(BOOL, p), INT)


def b_signbit(g, args, targs):
    t = ftype(g, args[0])
    x = reg_in(g, args[0], t)
    if t is FLOAT:
        b = g.fn.reg('r')
        g.emit('mov.b32 %s, %s;' % (b, x.r))
        r = g.fn.reg('r')
        g.emit('shr.u32 %s, %s, 31;' % (r, b))
        return V(INT, r)
    b = g.fn.reg('rd')
    g.emit('mov.b64 %s, %s;' % (b, x.r))
    r = g.fn.reg('rd')
    g.emit('shr.u64 %s, %s, 63;' % (r, b))
    return g.convert(V(ULLONG, r), INT)


def b_copysign(fixed=None):
    def h(g, args, targs):
        t = fixed or (FLOAT if ftype(g, args[0]) is FLOAT and ftype(g, args[1]) is FLOAT else DOUBLE)
        x, y = reg_in(g, args[0], t), reg_in(g, args[1], t)
        bits = t.size * 8
        cls = 'r' if bits == 32 else 'rd'
        a, b = g.fn.reg(cls), g.fn.reg(cls)
        g.emit('mov.b%d %s, %s;' % (bits, a, x.r))
        g.emit('mov.b%d %s, %s;' % (bits, b, y.r))
        m = (1 << (bits - 1))
        g.emit('and.b%d %s, %s, %d;' % (bits, a, a, m - 1))
        g.emit('and.b%d %s, %s, %d;' % (bits, b, b, m))
        g.emit('or.b%d %s, %s, %s;' % (bits, a, a, b))
        r = g.reg(t)
        g.emit('mov.b%d %s, %s;' % (bits, r, a))
        return V(t, r)
    return h


def generic_prelude(fname32, fname64):
    """Generic math name (exp, log, pow, ...): calls the float or double version from the prelude."""
    def h(g, args, targs):
        t = FLOAT if all(ftype(g, a) is FLOAT for a in args if not _is_ptr_arg(g, a)) else DOUBLE
        name = fname32 if t is FLOAT else fname64
        other = BUILTINS.get(name)
        if other is not None and other is not h:
            return other(g, args, targs)
        fds = g.prelude.get(name)
        if not fds:
            g.err("math function '%s' is not available" % name)
        fd = g.resolve_overload(fds, args, name)
        return g.inline(fd, args)
    return h


def _is_ptr_arg(g, a):
    t = g.peek_type(a)
    return isinstance(t, TPtr)


# ---- integer min/max/abs ------------------------------------------------------------------
def b_minmax(op, fixed=None):
    def h(g, args, targs):
        a, b = sv(g, args[0]), sv(g, args[1])
        t = fixed or P.common_type(a.t, b.t)
        a, b = g.convert(a, t), g.convert(b, t)
        if a.is_imm and b.is_imm:
            return V(t, imm=(min if op == 'min' else max)(a.imm, b.imm))
        a = g.in_reg(a)
        r = g.reg(t)
        g.emit('%s.%s %s, %s, %s;' % (op, P.op_type(t), r, a.r, g.operand(b)))
        return V(t, r)
    return h


def b_abs(fixed=None):
    def h(g, args, targs):
        v = sv(g, args[0])
        t = fixed or (v.t if v.t.is_float() else P.promote(v.t))
        v = g.in_reg(g.convert(v, t))
        r = g.reg(t)
        ty = P.op_type(t) if t.is_float() else ('s64' if t.size == 8 else 's32')
        g.emit('abs.%s %s, %s;' % (ty, r, v.r))
        return V(t, r)
    return h


# ---- vectors --------------------------------------------------------------------------------
def make_vec(tname):
    vt = VEC[tname]

    def h(g, args, targs):
        n = len(vt.fields)
        if len(args) != n:
            g.err('make_%s takes %d arguments' % (tname, n))
        leaves = {}
        for (fname, ft, fo), a in zip(vt.fields, args):
            leaves[(fname,)] = g.convert(sv(g, a), ft)
        return P.A(vt, leaves)
    return h


# ---- memory helpers -------------------------------------------------------------------------
def b_ldg(cache='nc'):
    def h(g, args, targs):
        pv = sv(g, args[0])
        if not isinstance(pv.t, TPtr):
            g.err('__ldg on a non-pointer')
        t = unqual(pv.t.to)
        pv = g.in_reg(pv)
        r0 = g.fn.reg('rd')
        g.emit('cvta.to.global.u64 %s, %s;' % (r0, pv.r))
        lv = P.LMem(t, 'global', r0)
        if isinstance(t, (TStruct, TArr)):
            leaves = {}
            for path, st, off in P.leaves_of(t):
                r = g.reg(st)
                g.emit('ld.global.%s.%s %s, %s;' % (cache, P.mem_type(st), r, g.addr_text(lv, off)))
                leaves[path] = V(st, r)
            return P.A(t, leaves)
        r = g.reg(t)
        g.emit('ld.global.%s.%s %s, %s;' % (cache, P.mem_type(t), r, g.addr_text(lv)))
        return V(t, r, space='generic' if isinstance(t, TPtr) else None)
    return h


def b_st_hint(g, args, targs):
    pv = sv(g, args[0])
    lv = g.deref(pv)
    g.store(lv, g.convert(g.rv(args[1]), unqual(lv.t)) if not isinstance(unqual(lv.t), TStruct) else g.rv(args[1]))
    return ret_void()


def b_isspace(space):
    def h(g, args, targs):
        pv = g.in_reg(sv(g, args[0]))
        p = g.preg()
        g.emit('isspacep.%s %s, %s;' % (space, p, pv.r))
        return g.convert(V(BOOL, p), UINT)
    return h


def b_cvta_shared(g, args, targs):
    pv = g.in_reg(sv(g, args[0]))
    r = g.fn.reg('rd')
    g.emit('cvta.to.shared.u64 %s, %s;' % (r, pv.r))
    return V(ULONG, r)


def b_clock(bits):
    def h(g, args, targs):
        if bits == 32:
            r = g.fn.reg('r')
            g.emit('mov.u32 %s, %%clock;' % r)
            return g.convert(V(INT, r), LONG)
        r = g.fn.reg('rd')
        g.emit('mov.u64 %s, %%clock64;' % r)
        return V(LLONG, r)
    return h


def b_globaltimer(g, args, targs):
    r = g.fn.reg('rd')
    g.emit('mov.u64 %s, %%globaltimer;' % r)
    return V(ULLONG, r)


# ---- printf / malloc -------------------------------------------------------------------------
def b_printf(g, args, targs):
    g.m.declare('vprintf', '.extern .func (.param .b32 rv) vprintf (.param .b64 fmt, .param .b64 args);')
    fmt = g.in_reg(sv(g, args[0]))
    vals = []
    for a in args[1:]:
        v = g.rv(a)
        if not isinstance(v, P.V):
            g.err('printf of an aggregate')
        t = v.t
        if t is FLOAT:
            v = g.convert(v, DOUBLE)
        elif t is BOOL or (t.is_int() and t.size < 4):
            v = g.convert(v, INT)
        vals.append(g.in_reg(v))
    if vals:
        off = 0
        layout = []
        for v in vals:
            sz = 8 if (isinstance(v.t, TPtr) or v.t.size == 8) else 4
            off = (off + sz - 1) // sz * sz
            layout.append((off, v, sz))
            off += sz
        base = g.fn.slot(off, 8)
        for o, v, sz in layout:
            mt = P.mem_type(v.t) if not isinstance(v.t, TPtr) else 'u64'
            g.emit('st.local.%s [__pk_depot+%d], %s;' % (mt, base + o, v.r))
        a0 = g.fn.reg('rd')
        g.emit('mov.u64 %s, __pk_depot;' % a0)
        a1 = g.fn.reg('rd')
        g.emit('add.s64 %s, %s, %d;' % (a1, a0, base))
        argp = g.fn.reg('rd')
        g.emit('cvta.local.u64 %s, %s;' % (argp, a1))
    else:
        argp = '0'
    r = g.fn.reg('r')
    g.emit('{')
    g.emit('.param .b64 pk_pf0;')
    g.emit('.param .b64 pk_pf1;')
    g.emit('.param .b32 pk_pfr;')
    g.emit('st.param.b64 [pk_pf0], %s;' % fmt.r)
    g.emit('st.param.b64 [pk_pf1], %s;' % argp)
    g.emit('call.uni (pk_pfr), vprintf, (pk_pf0, pk_pf1);')
    g.emit('ld.param.b32 %s, [pk_pfr];' % r)
    g.emit('}')
    return V(INT, r)


def b_malloc(g, args, targs):
    g.m.declare('malloc', '.extern .func (.param .b64 rv) malloc (.param .b64 size);')
    n = g.in_reg(cv(g, args[0], ULONG))
    r = g.fn.reg('rd')
    g.emit('{')
    g.emit('.param .b64 pk_ma0;')
    g.emit('.param .b64 pk_mar;')
    g.emit('st.param.b64 [pk_ma0], %s;' % n.r)
    g.emit('call.uni (pk_mar), malloc, (pk_ma0);')
    g.emit('ld.param.b64 %s, [pk_mar];' % r)
    g.emit('}')
    return V(TPtr(VOID), r, space='generic')


def b_free(g, args, targs):
    g.m.declare('free', '.extern .func free (.param .b64 ptr);')
    p = g.in_reg(sv(g, args[0]))
    g.emit('{')
    g.emit('.param .b64 pk_fr0;')
    g.emit('st.param.b64 [pk_fr0], %s;' % p.r)
    g.emit('call.uni free, (pk_fr0);')
    g.emit('}')
    return ret_void()


def b_memcpy(g, args, targs):
    d = g.in_reg(sv(g, args[0]))
    s = g.in_reg(sv(g, args[1]))
    n = cv(g, args[2], ULONG)
    if not n.is_imm:
        g.err('memcpy with a run-time size is not supported')
    dl = g.deref(P.V(TPtr(CHAR), d.r, space=d.space), CHAR)
    sl = g.deref(P.V(TPtr(CHAR), s.r, space=s.space), CHAR)
    for i in range(int(n.imm)):
        v = g.load_scalar(sl, CHAR, i)
        g.store_scalar(dl, CHAR, i, v)
    return d


def b_memset(g, args, targs):
    d = g.in_reg(sv(g, args[0]))
    v = cv(g, args[1], CHAR)
    n = cv(g, args[2], ULONG)
    if not n.is_imm:
        g.err('memset with a run-time size is not supported')
    dl = g.deref(P.V(TPtr(CHAR), d.r, space=d.space), CHAR)
    for i in range(int(n.imm)):
        g.store_scalar(dl, CHAR, i, v)
    return d


# ---- cooperative groups ------------------------------------------------------------------
def linear_tid(g):
    tx, ty, tz = (g.special('threadIdx', c) for c in 'xyz')
    nx, ny = (g.special('blockDim', c) for c in 'xy')
    r1 = g.fn.reg('r')
    g.emit('mad.lo.u32 %s, %s, %s, %s;' % (r1, tz.r, ny.r, ty.r))
    r2 = g.fn.reg('r')
    g.emit('mad.lo.u32 %s, %s, %s, %s;' % (r2, r1, nx.r, tx.r))
    return V(UINT, r2)


def block_size(g):
    nx, ny, nz = (g.special('blockDim', c) for c in 'xyz')
    r1 = g.fn.reg('r')
    g.emit('mul.lo.u32 %s, %s, %s;' % (r1, nx.r, ny.r))
    r2 = g.fn.reg('r')
    g.emit('mul.lo.u32 %s, %s, %s;' % (r2, r1, nz.r))
    return V(UINT, r2)


def b_this_thread_block(g, args, targs):
    return P.GroupV(TGroup('block'))


def b_this_grid(g, args, targs):
    return P.GroupV(TGroup('grid'))


def grid_block_rank(g):
    bx, by, bz = (g.special('blockIdx', c) for c in 'xyz')
    gx, gy = g.special('gridDim', 'x'), g.special('gridDim', 'y')
    r1 = g.fn.reg('r')
    g.emit('mad.lo.u32 %s, %s, %s, %s;' % (r1, bz.r, gy.r, by.r))
    r2 = g.fn.reg('r')
    g.emit('mad.lo.u32 %s, %s, %s, %s;' % (r2, r1, gx.r, bx.r))
    return V(UINT, r2)


def grid_num_blocks(g):
    gx, gy, gz = (g.special('gridDim', c) for c in 'xyz')
    r1 = g.fn.reg('r')
    g.emit('mul.lo.u32 %s, %s, %s;' % (r1, gx.r, gy.r))
    r2 = g.fn.reg('r')
    g.emit('mul.lo.u32 %s, %s, %s;' % (r2, r1, gz.r))
    return V(UINT, r2)


def grid_method(g, name):
    """grid_group queries (no grid-wide sync: that needs a cooperative launch)."""
    if name == 'thread_rank':
        blk, bs, lt = grid_block_rank(g), block_size(g), linear_tid(g)
        r, t = g.fn.reg('rd'), g.fn.reg('rd')
        g.emit('mul.wide.u32 %s, %s, %s;' % (r, blk.r, bs.r))
        g.emit('cvt.u64.u32 %s, %s;' % (t, lt.r))
        g.emit('add.u64 %s, %s, %s;' % (r, r, t))
        return V(ULLONG, r)
    if name in ('size', 'num_threads'):
        nb, bs = grid_num_blocks(g), block_size(g)
        r = g.fn.reg('rd')
        g.emit('mul.wide.u32 %s, %s, %s;' % (r, nb.r, bs.r))
        return V(ULLONG, r)
    if name == 'block_rank':
        return grid_block_rank(g)
    if name == 'num_blocks':
        return grid_num_blocks(g)
    if name == 'is_valid':
        return V(INT, imm=1)
    if name in ('dim_blocks', 'block_index', 'group_dim'):
        which = {'dim_blocks': 'gridDim', 'block_index': 'blockIdx', 'group_dim': 'gridDim'}[name]
        return P.A(VEC['uint3'], dict(((c,), g.special(which, c)) for c in 'xyz'))
    if name == 'sync':
        g.err('grid_group::sync needs a cooperative launch (not supported yet)')
    g.err('cooperative_groups: grid_group::%s is not supported' % name)


def b_tiled_partition(g, args, targs):
    if targs:
        n = g.env.const_eval(targs[0]) if not isinstance(targs[0], int) else targs[0]
    elif len(args) > 1:
        nv = cv(g, args[1], UINT)
        if not nv.is_imm:
            g.err('tiled_partition with a run-time size is not supported')
        n = int(nv.imm)
    else:
        g.err('tiled_partition needs a size')
    if n < 1 or n > 32 or n & (n - 1):
        g.err('tile size %d is not supported (powers of two up to 32)' % n)
    g.rv(args[0])
    return P.GroupV(TGroup('tile', n))


def b_coalesced_threads(g, args, targs):
    r = g.fn.reg('r')
    g.emit('activemask.b32 %s;' % r)
    return P.GroupV(TGroup('coalesced'), mask=V(UINT, r))


def match_any_mask(g, parent, label):
    """Lanes of the member mask `parent` whose label equals this lane's label (label: AST node or V)."""
    lab = match_operand(g, label if isinstance(label, P.V) else sv(g, label))
    bits = 64 if lab.t.size == 8 else 32
    pm = g.in_reg(parent)
    if g.m.target >= 70:
        r = g.fn.reg('r')
        g.emit('match.any.sync.b%d %s, %s, %s;' % (bits, r, lab.r, pm.r))
        return V(UINT, r)
    # before sm_70 there is no match instruction: take one label at a time, lowest remaining lane first.
    # `todo` is the same in every member lane, so all members run the loop the same number of times.
    todo, res, low, lead, m = (g.fn.reg('r') for _ in range(5))
    g.emit('mov.b32 %s, %s;' % (todo, pm.r))
    g.emit('mov.b32 %s, 0;' % res)
    top = g.fn.label()
    g.fn.place(top)
    g.emit('neg.s32 %s, %s;' % (low, todo))
    g.emit('and.b32 %s, %s, %s;' % (low, low, todo))
    g.emit('bfind.u32 %s, %s;' % (lead, low))
    other = do_shfl(g, 'idx', pm, lab, V(UINT, lead), '31')
    p = g.preg()
    g.emit('setp.eq.b%d %s, %s, %s;' % (bits, p, lab.r, g.in_reg(other).r))
    g.emit('vote.sync.ballot.b32 %s, %s, %s;' % (m, p, pm.r))
    g.emit('and.b32 %s, %s, %s;' % (m, m, todo))
    g.emit('@%s mov.b32 %s, %s;' % (p, res, m))
    g.emit('not.b32 %s, %s;' % (m, m))
    g.emit('and.b32 %s, %s, %s;' % (todo, todo, m))
    q = g.preg()
    g.emit('setp.ne.u32 %s, %s, 0;' % (q, todo))
    g.emit('@%s bra %s;' % (q, top))
    return V(UINT, res)


def partition_parent(g, arg, what):
    grp = g.rv(arg)
    if not isinstance(grp, P.GroupV) or grp.t.gkind not in ('tile', 'coalesced'):
        g.err('%s needs a thread_block_tile or coalesced_group' % what)
    return tile_mask(g, grp)


def b_labeled_partition(g, args, targs):
    parent = partition_parent(g, args[0], 'labeled_partition')
    return P.GroupV(TGroup('coalesced'), mask=match_any_mask(g, parent, args[1]))


def b_binary_partition(g, args, targs):
    parent = g.in_reg(partition_parent(g, args[0], 'binary_partition'))
    p = g.to_pred(sv(g, args[1]))
    b, nb, r = g.fn.reg('r'), g.fn.reg('r'), g.fn.reg('r')
    g.emit('vote.sync.ballot.b32 %s, %s, %s;' % (b, p, parent.r))
    g.emit('not.b32 %s, %s;' % (nb, b))
    g.emit('and.b32 %s, %s, %s;' % (nb, nb, parent.r))
    g.emit('selp.b32 %s, %s, %s, %s;' % (r, b, nb, p))
    return P.GroupV(TGroup('coalesced'), mask=V(UINT, r))


def coalesced_rank(g, mask):
    lt = b_lanemask('lt')(g, [], None)
    m = g.fn.reg('r')
    g.emit('and.b32 %s, %s, %s;' % (m, lt.r, g.operand(mask)))
    r = g.fn.reg('r')
    g.emit('popc.b32 %s, %s;' % (r, m))
    return r


def coalesced_shfl_rel(g, mask, val, delta, up):
    """coalesced_group::shfl_up/shfl_down: move by `delta` ranks; out-of-range ranks keep their own value."""
    mask = g.in_reg(mask)
    rank = coalesced_rank(g, mask)
    d = g.in_reg(cv(g, delta, UINT))
    src, size, lane, own, sel = (g.fn.reg('r') for _ in range(5))
    p = g.preg()
    if up:
        g.emit('setp.ge.u32 %s, %s, %s;' % (p, rank, d.r))
        g.emit('sub.u32 %s, %s, %s;' % (src, rank, d.r))
    else:
        g.emit('add.u32 %s, %s, %s;' % (src, rank, d.r))
        g.emit('popc.b32 %s, %s;' % (size, mask.r))
        g.emit('setp.lt.u32 %s, %s, %s;' % (p, src, size))
    g.emit('add.u32 %s, %s, 1;' % (src, src))
    g.emit('fns.b32 %s, %s, 0, %s;' % (lane, mask.r, src))
    g.emit('mov.u32 %s, %%laneid;' % own)
    g.emit('selp.b32 %s, %s, %s, %s;' % (sel, lane, own, p))
    return do_shfl(g, 'idx', mask, val, V(UINT, sel), '31')


def compact_ballot(g, mask, ballot):
    """Ballot bits of a coalesced group in rank order (bit k = k-th member)."""
    mask = g.in_reg(mask)
    r, k, i, t, u = (g.fn.reg('r') for _ in range(5))
    g.emit('mov.b32 %s, 0;' % r)
    g.emit('mov.b32 %s, 0;' % k)
    g.emit('mov.b32 %s, 0;' % i)
    top = g.fn.label()
    g.fn.place(top)
    pm, pl = g.preg(), g.preg()
    g.emit('shr.u32 %s, %s, %s;' % (t, mask.r, i))
    g.emit('and.b32 %s, %s, 1;' % (t, t))
    g.emit('setp.ne.u32 %s, %s, 0;' % (pm, t))
    g.emit('shr.u32 %s, %s, %s;' % (u, ballot, i))
    g.emit('and.b32 %s, %s, 1;' % (u, u))
    g.emit('shl.b32 %s, %s, %s;' % (u, u, k))
    g.emit('@%s or.b32 %s, %s, %s;' % (pm, r, r, u))
    g.emit('@%s add.u32 %s, %s, 1;' % (pm, k, k))
    g.emit('add.u32 %s, %s, 1;' % (i, i))
    g.emit('setp.lt.u32 %s, %s, 32;' % (pl, i))
    g.emit('@%s bra %s;' % (pl, top))
    return r


def b_cg_sync(g, args, targs):
    grp = g.rv(args[0])
    return group_method(g, grp, 'sync', [], None)


def tile_mask(g, grp):
    gk = grp.t.gkind
    if gk == 'coalesced':
        return grp.mask
    if gk in ('block', 'group'):
        return V(UINT, imm=0xffffffff)
    n = grp.t.n
    if n == 32:
        return V(UINT, imm=0xffffffff)
    lane = b_laneid(g, [], None)
    base = g.fn.reg('r')
    g.emit('and.b32 %s, %s, %d;' % (base, lane.r, (~(n - 1)) & 0xffffffff))
    r = g.fn.reg('r')
    g.emit('shl.b32 %s, %d, %s;' % (r, (1 << n) - 1, base))
    return V(UINT, r)


def apply_op(g, op, a, b):
    t = a.t
    if isinstance(op, TFunctor):
        o = op.op
    else:
        o = op
    if o == 'plus':
        return g.arith('+', a, b)
    if o in ('less', 'min'):
        r = g.reg(t)
        g.emit('min.%s %s, %s, %s;' % (P.op_type(t), r, g.in_reg(a).r, g.operand(b)))
        return V(t, r)
    if o in ('greater', 'max'):
        r = g.reg(t)
        g.emit('max.%s %s, %s, %s;' % (P.op_type(t), r, g.in_reg(a).r, g.operand(b)))
        return V(t, r)
    if o in ('bit_and', 'bit_or', 'bit_xor'):
        return g.arith({'bit_and': '&', 'bit_or': '|', 'bit_xor': '^'}[o], a, b)
    g.err('unsupported reduction operator')


def selp_type(t):
    return 'f32' if t is FLOAT else ('f64' if t is DOUBLE else ('b64' if t.size == 8 else 'b32'))


def coalesced_reduce(g, mask, v, op):
    """cg::reduce over a coalesced group (any member mask): a rank-ordered tree, then a broadcast from rank 0."""
    mask = g.in_reg(mask)
    rank = coalesced_rank(g, mask)
    size, own = g.fn.reg('r'), g.fn.reg('r')
    g.emit('popc.b32 %s, %s;' % (size, mask.r))
    g.emit('mov.u32 %s, %%laneid;' % own)
    acc = g.reg(v.t)
    g.mov_to(acc, g.in_reg(v))
    for off in (16, 8, 4, 2, 1):
        src, lane, sel = g.fn.reg('r'), g.fn.reg('r'), g.fn.reg('r')
        p = g.preg()
        g.emit('add.u32 %s, %s, %d;' % (src, rank, off))
        g.emit('setp.lt.u32 %s, %s, %s;' % (p, src, size))
        g.emit('add.u32 %s, %s, 1;' % (src, src))
        g.emit('fns.b32 %s, %s, 0, %s;' % (lane, mask.r, src))
        g.emit('selp.b32 %s, %s, %s, %s;' % (sel, lane, own, p))
        other = do_shfl(g, 'idx', mask, V(v.t, acc), V(UINT, sel), '31')
        r = g.in_reg(apply_op(g, op, V(v.t, acc), other))
        g.emit('selp.%s %s, %s, %s, %s;' % (selp_type(v.t), acc, r.r, acc, p))
    lane0 = g.fn.reg('r')
    g.emit('fns.b32 %s, %s, 0, 1;' % (lane0, mask.r))
    return do_shfl(g, 'idx', mask, V(v.t, acc), V(UINT, lane0), '31')


def b_cg_reduce(g, args, targs):
    grp = g.rv(args[0])
    if not isinstance(grp, P.GroupV) or grp.t.gkind not in ('tile', 'coalesced'):
        g.err('cg::reduce is supported on thread_block_tile and coalesced_group')
    v = sv(g, args[1])
    if v.t.is_int():
        v = g.convert(v, P.promote(v.t))
    op = g.rv(args[2])
    if not isinstance(op, P.GroupV):
        g.err('cg::reduce needs cg::plus/less/greater/bit_* operators')
    if grp.t.gkind == 'coalesced':
        return coalesced_reduce(g, grp.mask, v, op.t)
    n = grp.t.n
    mask = tile_mask(g, grp)
    c = str(((32 - n) << 8) | 0x1f)
    o = n // 2
    while o >= 1:
        other = do_shfl(g, 'bfly', mask, v, V(UINT, imm=o), c)
        v = apply_op(g, op.t, g.in_reg(v), other)
        o //= 2
    return v


def b_cg_scan(inclusive):
    def h(g, args, targs):
        grp = g.rv(args[0])
        if not isinstance(grp, P.GroupV) or grp.t.gkind != 'tile':
            g.err('cg scans are supported on thread_block_tile only')
        v = sv(g, args[1])
        if v.t.is_int():
            v = g.convert(v, P.promote(v.t))
        op = g.rv(args[2]).t if len(args) > 2 else TFunctor('plus', v.t)
        n = grp.t.n
        mask = tile_mask(g, grp)
        lane = b_laneid(g, [], None)
        rank = g.fn.reg('r')
        g.emit('and.b32 %s, %s, %d;' % (rank, lane.r, n - 1))
        c = str((32 - n) << 8)
        orig = g.in_reg(v)
        acc = g.reg(v.t)
        g.mov_to(acc, orig)
        o = 1
        while o < n:
            other = do_shfl(g, 'up', mask, V(v.t, acc), V(UINT, imm=o), c)
            s = apply_op(g, op, V(v.t, acc), other)
            p = g.preg()
            g.emit('setp.ge.u32 %s, %s, %d;' % (p, rank, o))
            st = 'f32' if v.t is FLOAT else ('f64' if v.t is DOUBLE else ('b64' if v.t.size == 8 else 'b32'))
            g.emit('selp.%s %s, %s, %s, %s;' % (st, acc, g.in_reg(s).r, acc, p))
            o *= 2
        if inclusive:
            return V(v.t, acc)
        # exclusive: shift up by one, rank 0 gets the identity (0 for plus)
        prev = do_shfl(g, 'up', mask, V(v.t, acc), V(UINT, imm=1), c)
        p = g.preg()
        g.emit('setp.eq.u32 %s, %s, 0;' % (p, rank))
        r = g.reg(v.t)
        st = 'f32' if v.t is FLOAT else ('f64' if v.t is DOUBLE else ('b64' if v.t.size == 8 else 'b32'))
        zero = P.fmt_imm(0.0 if v.t.is_float() else 0, v.t)
        g.emit('selp.%s %s, %s, %s, %s;' % (st, r, zero, g.in_reg(prev).r, p))
        return V(v.t, r)
    return h


def group_method(g, grp, name, args, targs):
    gk = grp.t.gkind if isinstance(grp.t, TGroup) else None
    if isinstance(grp.t, TFunctor):
        a, b = sv(g, args[0]), sv(g, args[1])
        t = P.common_type(a.t, b.t)
        return apply_op(g, grp.t, g.in_reg(g.convert(a, t)), g.convert(b, t))
    if gk == 'grid':
        return grid_method(g, name)
    if name == 'sync':
        if gk in ('block', 'group'):
            g.emit('bar.sync 0;')
        else:
            m = tile_mask(g, grp)
            g.emit('bar.warp.sync %s;' % g.operand(m))
        return ret_void()
    if name in ('thread_rank',):
        if gk in ('block', 'group'):
            return linear_tid(g)
        if gk == 'tile':
            lane = b_laneid(g, [], None)
            r = g.fn.reg('r')
            g.emit('and.b32 %s, %s, %d;' % (r, lane.r, grp.t.n - 1))
            return V(UINT, r)
        if gk == 'coalesced':
            lt = b_lanemask('lt')(g, [], None)
            m = g.fn.reg('r')
            g.emit('and.b32 %s, %s, %s;' % (m, lt.r, g.operand(grp.mask)))
            r = g.fn.reg('r')
            g.emit('popc.b32 %s, %s;' % (r, m))
            return V(UINT, r)
    if name in ('size', 'num_threads'):
        if gk in ('block', 'group'):
            return block_size(g)
        if gk == 'tile':
            return V(UINT, imm=grp.t.n)
        if gk == 'coalesced':
            r = g.fn.reg('r')
            g.emit('popc.b32 %s, %s;' % (r, g.in_reg(grp.mask).r))
            return V(UINT, r)
    if name in ('meta_group_rank', 'meta_group_size') and gk == 'tile':
        n = grp.t.n
        if name == 'meta_group_rank':
            lt = linear_tid(g)
            r = g.fn.reg('r')
            g.emit('shr.u32 %s, %s, %d;' % (r, lt.r, n.bit_length() - 1))
            return V(UINT, r)
        bs = block_size(g)
        r = g.fn.reg('r')
        g.emit('add.u32 %s, %s, %d;' % (r, bs.r, n - 1))
        g.emit('shr.u32 %s, %s, %d;' % (r, r, n.bit_length() - 1))
        return V(UINT, r)
    if name in ('meta_group_rank', 'meta_group_size') and gk == 'coalesced':
        return V(UINT, imm=0 if name == 'meta_group_rank' else 1)
    if gk == 'block' and name in ('group_index', 'thread_index', 'dim_threads', 'group_dim'):
        which = {'group_index': 'blockIdx', 'thread_index': 'threadIdx', 'dim_threads': 'blockDim',
                 'group_dim': 'blockDim'}[name]
        return P.A(VEC['uint3'], dict(((c,), g.special(which, c)) for c in 'xyz'))
    if gk in ('tile', 'coalesced') and name in ('shfl', 'shfl_up', 'shfl_down', 'shfl_xor'):
        mode = {'shfl': 'idx', 'shfl_up': 'up', 'shfl_down': 'down', 'shfl_xor': 'bfly'}[name]
        val = sv(g, args[0])
        b = cv(g, args[1], UINT)
        mask = tile_mask(g, grp)
        if gk == 'coalesced':
            if mode in ('up', 'down'):
                return coalesced_shfl_rel(g, mask, val, args[1], mode == 'up')
            if mode != 'idx':
                g.err('coalesced_group::%s is not supported' % name)
            # rank -> lane: position of the (rank+1)-th set bit of the mask
            b1 = g.in_reg(g.arith('+', b, V(UINT, imm=1)))
            lane = g.fn.reg('r')
            g.emit('fns.b32 %s, %s, 0, %s;' % (lane, g.in_reg(mask).r, b1.r))
            return do_shfl(g, 'idx', mask, val, V(UINT, lane), '31')
        n = grp.t.n
        c = str(((32 - n) << 8) | (0 if mode == 'up' else 0x1f))
        return do_shfl(g, mode, mask, val, b, c)
    if gk in ('tile', 'coalesced') and name in ('any', 'all', 'ballot'):
        mask = tile_mask(g, grp)
        p = g.to_pred(sv(g, args[0]))
        if name == 'ballot':
            r = g.fn.reg('r')
            g.emit('vote.sync.ballot.b32 %s, %s, %s;' % (r, p, g.operand(mask)))
            if gk == 'coalesced':
                return V(UINT, compact_ballot(g, mask, r))
            if gk == 'tile' and grp.t.n < 32:
                lane = b_laneid(g, [], None)
                base = g.fn.reg('r')
                g.emit('and.b32 %s, %s, %d;' % (base, lane.r, (~(grp.t.n - 1)) & 0xffffffff))
                g.emit('shr.u32 %s, %s, %s;' % (r, r, base))
                g.emit('and.b32 %s, %s, %d;' % (r, r, (1 << grp.t.n) - 1))
            return V(UINT, r)
        q = g.preg()
        g.emit('vote.sync.%s.pred %s, %s, %s;' % (name, q, p, g.operand(mask)))
        return g.convert(V(BOOL, q), INT)
    if gk == 'tile' and name in ('match_any', 'match_all'):
        g.m.need_sm = max(g.m.need_sm, 70)
        g.err('thread_block_tile::%s is not supported yet' % name)
    g.err('cooperative_groups: %s::%s is not supported' % (gk, name))


# ---------------------------------------------------------------------------------------------
BUILTINS = {
    '__syncthreads': b_syncthreads,
    '__syncthreads_count': b_syncthreads_red('popc'),
    '__syncthreads_and': b_syncthreads_red('and'),
    '__syncthreads_or': b_syncthreads_red('or'),
    '__syncwarp': b_syncwarp,
    '__threadfence': b_fence('gl'),
    '__threadfence_block': b_fence('cta'),
    '__threadfence_system': b_fence('sys'),
    '__trap': b_trap,
    '__shfl_sync': shfl('idx'), '__shfl_up_sync': shfl('up'), '__shfl_down_sync': shfl('down'),
    '__shfl_xor_sync': shfl('bfly'),
    '__shfl': shfl('idx', True), '__shfl_up': shfl('up', True), '__shfl_down': shfl('down', True),
    '__shfl_xor': shfl('bfly', True),
    '__all_sync': vote('all'), '__any_sync': vote('any'), '__uni_sync': vote('uni'), '__ballot_sync': vote('ballot'),
    '__all': vote('all', True), '__any': vote('any', True), '__ballot': vote('ballot', True),
    '__activemask': b_activemask,
    '__lanemask_lt': b_lanemask('lt'), '__lanemask_le': b_lanemask('le'), '__lanemask_gt': b_lanemask('gt'),
    '__lanemask_ge': b_lanemask('ge'), '__lanemask_eq': b_lanemask('eq'), '__laneid': b_laneid,
    '__match_any_sync': b_match_any, '__match_all_sync': b_match_all,
    '__nanosleep': b_nanosleep,
    '__popc': b_popc(32), '__popcll': b_popc(64), '__clz': b_clz(32), '__clzll': b_clz(64),
    '__ffs': b_ffs(32), '__ffsll': b_ffs(64), '__brev': b_brev(32), '__brevll': b_brev(64),
    '__byte_perm': b_byte_perm,
    '__funnelshift_l': b_funnel('l', 'wrap'), '__funnelshift_lc': b_funnel('l', 'clamp'),
    '__funnelshift_r': b_funnel('r', 'wrap'), '__funnelshift_rc': b_funnel('r', 'clamp'),
    '__mul24': b_mulx('mul24.lo.s32', INT), '__umul24': b_mulx('mul24.lo.u32', UINT),
    '__mulhi': b_mulx('mul.hi.s32', INT), '__umulhi': b_mulx('mul.hi.u32', UINT),
    '__mul64hi': b_mulx('mul.hi.s64', LLONG), '__umul64hi': b_mulx('mul.hi.u64', ULLONG),
    '__sad': b_sad(INT), '__usad': b_sad(UINT),
    '__hadd': b_hadd(True, False), '__rhadd': b_hadd(True, True), '__uhadd': b_hadd(False, False),
    '__urhadd': b_hadd(False, True),
    '__float_as_int': b_bitcast(FLOAT, INT), '__int_as_float': b_bitcast(INT, FLOAT),
    '__float_as_uint': b_bitcast(FLOAT, UINT), '__uint_as_float': b_bitcast(UINT, FLOAT),
    '__double_as_longlong': b_bitcast(DOUBLE, LLONG), '__longlong_as_double': b_bitcast(LLONG, DOUBLE),
    '__double2hiint': b_double2hi(True), '__double2loint': b_double2hi(False),
    '__hiloint2double': b_hiloint2double,
    'sqrtf': f_unary('sqrt.rn.f32', fixed=FLOAT), 'sqrt': f_unary('sqrt.rn.f32', 'sqrt.rn.f64'),
    'rsqrtf': f_unary('rsqrt.approx.f32', fixed=FLOAT),
    'fabsf': f_unary('abs.f32', fixed=FLOAT), 'fabs': f_unary('abs.f32', 'abs.f64'),
    'floorf': f_unary('cvt.rmi.f32.f32', fixed=FLOAT), 'floor': f_unary('cvt.rmi.f32.f32', 'cvt.rmi.f64.f64'),
    'ceilf': f_unary('cvt.rpi.f32.f32', fixed=FLOAT), 'ceil': f_unary('cvt.rpi.f32.f32', 'cvt.rpi.f64.f64'),
    'truncf': f_unary('cvt.rzi.f32.f32', fixed=FLOAT), 'trunc': f_unary('cvt.rzi.f32.f32', 'cvt.rzi.f64.f64'),
    'rintf': f_unary('cvt.rni.f32.f32', fixed=FLOAT), 'rint': f_unary('cvt.rni.f32.f32', 'cvt.rni.f64.f64'),
    'nearbyintf': f_unary('cvt.rni.f32.f32', fixed=FLOAT),
    'nearbyint': f_unary('cvt.rni.f32.f32', 'cvt.rni.f64.f64'),
    'fminf': f_binary('min.f32', fixed=FLOAT), 'fmin': f_binary('min.f32', 'min.f64'),
    'fmaxf': f_binary('max.f32', fixed=FLOAT), 'fmax': f_binary('max.f32', 'max.f64'),
    'fmaf': f_fma(fixed=FLOAT), 'fma': f_fma(),
    'exp2f': f_unary('ex2.approx.f32', fixed=FLOAT),
    '__expf': b_fast('exp'), '__exp10f': b_fast('exp10'), '__logf': b_fast('log'), '__log2f': b_fast('log2'),
    '__log10f': b_fast('log10'), '__sinf': b_fast('sin'), '__cosf': b_fast('cos'), '__tanf': b_fast('tan'),
    '__powf': b_powf_fast, '__sincosf': b_sincosf_fast,
    '__saturatef': b_saturate, '__fdividef': b_fdividef,
    '__frcp_rn': b_rounded('rcp', FLOAT, 'rn'), '__fsqrt_rn': b_rounded('sqrt', FLOAT, 'rn'),
    '__drcp_rn': b_rounded('rcp', DOUBLE, 'rn'), '__dsqrt_rn': b_rounded('sqrt', DOUBLE, 'rn'),
    'isnan': b_isnan, 'isinf': b_isinf, 'isfinite': b_isfinite, 'signbit': b_signbit,
    '__isnanf': b_isnan, '__isinff': b_isinf, '__finitef': b_isfinite, '__isnan': b_isnan, '__isinf': b_isinf,
    'copysignf': b_copysign(FLOAT), 'copysign': b_copysign(),
    'min': b_minmax('min'), 'max': b_minmax('max'), 'umin': b_minmax('min', UINT), 'umax': b_minmax('max', UINT),
    'llmin': b_minmax('min', LLONG), 'llmax': b_minmax('max', LLONG), 'ullmin': b_minmax('min', ULLONG),
    'ullmax': b_minmax('max', ULLONG),
    'abs': b_abs(), 'labs': b_abs(LONG), 'llabs': b_abs(LLONG),
    '__ldg': b_ldg('nc'), '__ldcg': b_ldg('cg'), '__ldca': b_ldg('ca'), '__ldcs': b_ldg('cs'),
    '__ldlu': b_ldg('lu'), '__ldcv': b_ldg('cv'),
    '__stcg': b_st_hint, '__stcs': b_st_hint, '__stwb': b_st_hint, '__stwt': b_st_hint,
    '__isGlobal': b_isspace('global'), '__isShared': b_isspace('shared'), '__isLocal': b_isspace('local'),
    '__isConstant': b_isspace('const'), '__cvta_generic_to_shared': b_cvta_shared,
    'clock': b_clock(32), 'clock64': b_clock(64), '__globaltimer': b_globaltimer,
    'printf': b_printf, 'malloc': b_malloc, 'free': b_free, 'memcpy': b_memcpy, 'memset': b_memset,
    'this_thread_block': b_this_thread_block, 'this_grid': b_this_grid, 'tiled_partition': b_tiled_partition,
    'coalesced_threads': b_coalesced_threads, 'sync': b_cg_sync, 'reduce': b_cg_reduce,
    'labeled_partition': b_labeled_partition, 'binary_partition': b_binary_partition,
    'inclusive_scan': b_cg_scan(True), 'exclusive_scan': b_cg_scan(False),
}

for _op, _nm in (('add', 'Add'), ('sub', 'Sub'), ('exch', 'Exch'), ('min', 'Min'), ('max', 'Max'), ('inc', 'Inc'),
                 ('dec', 'Dec'), ('and', 'And'), ('or', 'Or'), ('xor', 'Xor'), ('cas', 'CAS')):
    BUILTINS['atomic' + _nm] = atomic(_op)
    BUILTINS['atomic' + _nm + '_block'] = atomic(_op, 'block')
    BUILTINS['atomic' + _nm + '_system'] = atomic(_op, 'system')

for _r in ('rn', 'rz', 'ru', 'rd'):
    _pr = {'rn': 'rn', 'rz': 'rz', 'ru': 'rp', 'rd': 'rm'}[_r]
    for _o in ('add', 'sub', 'mul', 'div'):
        BUILTINS['__f%s_%s' % (_o, _r)] = b_rounded(_o, FLOAT, _pr)
        BUILTINS['__d%s_%s' % (_o, _r)] = b_rounded(_o, DOUBLE, _pr)
    BUILTINS['__fmaf_' + _r] = b_rounded('fma', FLOAT, _pr)
    BUILTINS['__fma_' + _r] = b_rounded('fma', DOUBLE, _pr)
    for _s in CONV_T:
        for _d in CONV_T:
            if _s != _d and not (_s in ('int', 'uint', 'll', 'ull') and _d in ('int', 'uint', 'll', 'ull')):
                BUILTINS['__%s2%s_%s' % (_s, _d, _r)] = conversion(_s, _d, _r)

for _tn in VEC:
    if _tn != 'dim3':
        BUILTINS['make_' + _tn] = make_vec(_tn)

# Math functions implemented in ptx_prelude.cu, reachable by their C names and C++ overloads.
for _f in ('exp', 'log', 'log2', 'log10', 'exp10', 'expm1', 'log1p', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
           'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'cbrt', 'erf', 'erfc', 'round', 'pow', 'atan2',
           'hypot', 'fmod', 'remainder', 'fdim', 'ldexp', 'scalbn', 'logb', 'ilogb', 'lgamma', 'tgamma', 'erfinv',
           'erfcinv', 'sinpi', 'cospi', 'rcbrt', 'normcdf', 'normcdfinv', 'lround', 'llround', 'lrint', 'llrint',
           'exp2', 'frexp', 'modf', 'nextafter'):
    BUILTINS.setdefault(_f, generic_prelude(_f + 'f', _f))
BUILTINS['rsqrt'] = generic_prelude('rsqrtf', 'rsqrt')
