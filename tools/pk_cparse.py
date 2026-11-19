"""Parser for CUDA device code: a C subset plus the C++ features kernels commonly use
(bool, references, casts, auto, namespaces, functional casts, brace initialisers).

Produces an AST and C types. Used by pk_ptx.py, PolyKernel's own CUDA -> PTX compiler.
Input is the token list of one top-level item (see pk_ocl.split_items); type names are
resolved through an environment object as items are parsed in source order.
"""
import re
import struct


class ParseError(Exception):
    pass


# ---------------------------------------------------------------------------
# C types
# ---------------------------------------------------------------------------
class CType:
    size = 0
    align = 1
    kind = '?'

    def is_int(self):
        return False

    def is_float(self):
        return False

    def is_arith(self):
        return self.is_int() or self.is_float()

    def is_ptr(self):
        return False

    def is_scalar(self):
        return self.is_arith() or self.is_ptr()

    def is_agg(self):
        return False

    def strip(self):
        return self


class TVoid(CType):
    kind = 'void'

    def __repr__(self):
        return 'void'


class TInt(CType):
    kind = 'int'

    def __init__(self, size, signed, name, rank):
        self.size = self.align = size
        self.signed = signed
        self.name = name
        self.rank = rank

    def is_int(self):
        return True

    def __repr__(self):
        return self.name


class TFloat(CType):
    kind = 'float'

    def __init__(self, size, name):
        self.size = self.align = size
        self.name = name

    def is_float(self):
        return True

    def __repr__(self):
        return self.name


class TPtr(CType):
    kind = 'ptr'
    size = align = 8

    def __init__(self, to, restrict=False):
        self.to = to
        self.restrict = restrict

    def is_ptr(self):
        return True

    def __repr__(self):
        return '%r*' % (self.to,)


class TRef(CType):
    kind = 'ref'
    size = align = 8

    def __init__(self, to):
        self.to = to

    def __repr__(self):
        return '%r&' % (self.to,)


class TArr(CType):
    kind = 'arr'

    def __init__(self, elem, n):
        self.elem = elem
        self.n = n
        self.align = elem.align
        self.size = elem.size * (n or 0)

    def is_agg(self):
        return True

    def __repr__(self):
        return '%r[%s]' % (self.elem, self.n if self.n is not None else '')


class TStruct(CType):
    kind = 'struct'

    def __init__(self, name, union=False):
        self.name = name
        self.union = union
        self.fields = []          # (name, type, offset)
        self.complete = False
        self.size = 0
        self.align = 1
        self.vec = None           # element type for CUDA vector types (float4, ...)
        self.methods = False      # has member functions (not supported for calls yet)

    def layout(self, members, min_align=1):
        off = 0
        al = max(1, min_align)
        out = []
        for name, t in members:
            a = t.align
            al = max(al, a)
            if self.union:
                out.append((name, t, 0))
                off = max(off, t.size)
            else:
                off = (off + a - 1) // a * a
                out.append((name, t, off))
                off += t.size
        self.fields = out
        self.align = al
        self.size = (off + al - 1) // al * al
        self.complete = True

    def field(self, name):
        for f in self.fields:
            if f[0] == name:
                return f
        for f in self.fields:   # anonymous struct/union members
            if f[0] is None and isinstance(f[1], TStruct):
                sub = f[1].field(name)
                if sub:
                    return (sub[0], sub[1], f[2] + sub[2])
        return None

    def is_agg(self):
        return True

    def __repr__(self):
        return self.name or '<anon struct>'


class TFunc(CType):
    kind = 'func'

    def __init__(self, ret, params, variadic=False):
        self.ret = ret
        self.params = params
        self.variadic = variadic

    def __repr__(self):
        return '%r(%s)' % (self.ret, ', '.join(map(repr, self.params)))


class TGroup(CType):
    """A cooperative_groups handle (thread_block, thread_block_tile<N>, coalesced_group, grid_group)."""
    kind = 'group'
    size = align = 4

    def __init__(self, gkind, n=0):
        self.gkind = gkind
        self.n = n

    def __repr__(self):
        return 'cg::%s%s' % (self.gkind, '<%d>' % self.n if self.n else '')


class TFunctor(CType):
    """cooperative_groups::plus<T>() and friends."""
    kind = 'functor'
    size = align = 1

    def __init__(self, op, t):
        self.op = op
        self.t = t

    def __repr__(self):
        return 'cg::%s<%r>' % (self.op, self.t)


VOID = TVoid()
BOOL = TInt(1, False, 'bool', 0)
CHAR = TInt(1, True, 'char', 1)
SCHAR = TInt(1, True, 'signed char', 1)
UCHAR = TInt(1, False, 'unsigned char', 1)
SHORT = TInt(2, True, 'short', 2)
USHORT = TInt(2, False, 'unsigned short', 2)
INT = TInt(4, True, 'int', 3)
UINT = TInt(4, False, 'unsigned int', 3)
LONG = TInt(8, True, 'long', 4)
ULONG = TInt(8, False, 'unsigned long', 4)
LLONG = TInt(8, True, 'long long', 5)
ULLONG = TInt(8, False, 'unsigned long long', 5)
FLOAT = TFloat(4, 'float')
DOUBLE = TFloat(8, 'double')

TYPEDEFS = {
    'size_t': ULONG, 'ssize_t': LONG, 'ptrdiff_t': LONG, 'intptr_t': LONG, 'uintptr_t': ULONG,
    'int8_t': SCHAR, 'uint8_t': UCHAR, 'int16_t': SHORT, 'uint16_t': USHORT, 'int32_t': INT, 'uint32_t': UINT,
    'int64_t': LONG, 'uint64_t': ULONG, 'uint': UINT, 'uchar': UCHAR, 'ushort': USHORT, 'ulong': ULONG,
    'cudaError_t': INT, 'cudaStream_t': TPtr(VOID), '_Bool': BOOL, 'wchar_t': INT, 'char16_t': USHORT,
    'char32_t': UINT, 'int_least32_t': INT, 'uint_least32_t': UINT,
}


def vec_types():
    """CUDA's built-in vector types with the sizes and alignments of the programming guide."""
    out = {}
    table = [('char', SCHAR, (1, 2, 1, 4)), ('uchar', UCHAR, (1, 2, 1, 4)), ('short', SHORT, (2, 4, 2, 8)),
             ('ushort', USHORT, (2, 4, 2, 8)), ('int', INT, (4, 8, 4, 16)), ('uint', UINT, (4, 8, 4, 16)),
             ('long', LONG, (8, 16, 8, 16)), ('ulong', ULONG, (8, 16, 8, 16)), ('longlong', LLONG, (8, 16, 8, 16)),
             ('ulonglong', ULLONG, (8, 16, 8, 16)), ('float', FLOAT, (4, 8, 4, 16)),
             ('double', DOUBLE, (8, 16, 8, 16))]
    for base, et, aligns in table:
        for n in range(1, 5):
            t = TStruct(base + str(n))
            t.layout([(c, et) for c in 'xyzw'[:n]], aligns[n - 1])
            t.vec = et
            out[t.name] = t
    d = TStruct('dim3')
    d.layout([(c, UINT) for c in 'xyz'])
    d.vec = UINT
    out['dim3'] = d
    return out


VEC = vec_types()


def unqual(t):
    return t.to if isinstance(t, TRef) else t


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------
class Node:
    __slots__ = ()

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, ', '.join('%s=%r' % (k, getattr(self, k, None))
                                                         for k in self.__slots__))


def node(name, fields):
    def init(self, *a, **kw):
        for k, v in zip(fields, a):
            setattr(self, k, v)
        for k in fields[len(a):]:
            setattr(self, k, kw.get(k))
    return type(name, (Node,), {'__slots__': tuple(fields), '__init__': init})


ENum = node('ENum', ['v', 't'])                 # integer or floating literal
EStr = node('EStr', ['v'])                      # bytes
EName = node('EName', ['name', 'targs'])
EMember = node('EMember', ['obj', 'name', 'arrow'])
EIndex = node('EIndex', ['base', 'idx'])
ECall = node('ECall', ['fn', 'args'])
EUnary = node('EUnary', ['op', 'e'])
EBin = node('EBin', ['op', 'a', 'b'])
EAssign = node('EAssign', ['op', 'l', 'r'])
ECond = node('ECond', ['c', 'a', 'b'])
ECast = node('ECast', ['t', 'e', 'style'])     # e may be a list for functional casts with several args
ESizeof = node('ESizeof', ['t', 'e'])
EComma = node('EComma', ['a', 'b'])
EInit = node('EInit', ['items'])
EType = node('EType', ['t'])                    # a type used as a value (cg::plus<int>())
EAsm = node('EAsm', ['text', 'outs', 'ins'])

SBlock = node('SBlock', ['stmts'])
SExpr = node('SExpr', ['e'])
SDecl = node('SDecl', ['decls'])
SIf = node('SIf', ['c', 't', 'e'])
SWhile = node('SWhile', ['c', 'body'])
SDo = node('SDo', ['body', 'c'])
SFor = node('SFor', ['init', 'c', 'step', 'body', 'unroll'])
SSwitch = node('SSwitch', ['e', 'body'])
SCase = node('SCase', ['v'])
SDefault = node('SDefault', [])
SBreak = node('SBreak', [])
SContinue = node('SContinue', [])
SReturn = node('SReturn', ['e'])
SEmpty = node('SEmpty', [])


class VarDecl:
    _ids = 0

    def __init__(self, name, t, init=None, storage=()):
        VarDecl._ids += 1
        self.id = VarDecl._ids
        self.name = name
        self.t = t
        self.init = init
        self.storage = set(storage)

    def __repr__(self):
        return 'VarDecl(%s: %r)' % (self.name, self.t)


class FuncDef:
    def __init__(self, name, ns, ret, params, body, kind, storage):
        self.name = name
        self.ns = ns
        self.ret = ret
        self.params = params      # list of VarDecl (init = default argument)
        self.body = body          # SBlock or None for a prototype
        self.kind = kind          # 'kernel' | 'device' | 'host'
        self.storage = storage

    def __repr__(self):
        return 'FuncDef(%s)' % self.name


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------
_ESC = {'n': 10, 't': 9, 'r': 13, '0': 0, '\\': 92, "'": 39, '"': 34, 'a': 7, 'b': 8, 'f': 12, 'v': 11, '?': 63}


def unescape(body):
    out = bytearray()
    i = 0
    while i < len(body):
        c = body[i]
        if c != '\\':
            out += c.encode('utf-8')
            i += 1
            continue
        i += 1
        c = body[i]
        if c == 'x':
            j = i + 1
            while j < len(body) and body[j] in '0123456789abcdefABCDEF':
                j += 1
            out.append(int(body[i + 1:j], 16) & 0xff)
            i = j
        elif c in '01234567':
            j = i
            while j < len(body) and j < i + 3 and body[j] in '01234567':
                j += 1
            out.append(int(body[i:j], 8) & 0xff)
            i = j
        else:
            out.append(_ESC.get(c, ord(c)))
            i += 1
    return bytes(out)


def parse_number(s):
    t = s.replace("'", '')
    low = t.lower()
    is_hex = low.startswith('0x')
    if (('.' in low or ('e' in low and not is_hex) or ('p' in low and is_hex)) and not low.startswith('0b')):
        suf = ''
        while low and low[-1] in 'fl':
            suf = low[-1] + suf
            low = low[:-1]
        v = float.fromhex(low) if is_hex else float(low)
        if 'f' in suf:
            return v, FLOAT
        return v, DOUBLE
    m = re.match(r'^(0x[0-9a-f]+|0b[01]+|0[0-7]*|[1-9][0-9]*)([ul]*)$', low)
    if not m:
        raise ParseError('bad number %r' % s)
    digits, suf = m.groups()
    if digits.startswith('0x'):
        v = int(digits, 16)
    elif digits.startswith('0b'):
        v = int(digits[2:], 2)
    elif digits.startswith('0') and len(digits) > 1:
        v = int(digits, 8)
    else:
        v = int(digits)
    dec = not (digits.startswith('0') and len(digits) > 1)
    u = 'u' in suf
    ln = suf.count('l')
    if ln >= 2:
        cands = [ULLONG] if u else ([LLONG] if dec else [LLONG, ULLONG])
    elif ln == 1:
        cands = [ULONG] if u else ([LONG] if dec else [LONG, ULONG])
    else:
        cands = [UINT, ULONG] if u else ([INT, LONG] if dec else [INT, UINT, LONG, ULONG])
    for c in cands:
        if v < (1 << (c.size * 8 - (1 if c.signed else 0))):
            return v, c
    return v & ((1 << 64) - 1), ULLONG


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
SPECIFIERS = {'static', 'extern', 'inline', '__inline__', '__forceinline__', '__noinline__', '__device__',
              '__global__', '__host__', '__shared__', '__constant__', '__managed__', 'constexpr', 'register',
              'const', 'volatile', '__restrict__', '__restrict', 'restrict', 'thread_local', 'mutable',
              '__grid_constant__', 'typename', 'explicit', 'virtual', 'friend'}
BASE_WORDS = {'void', 'bool', '_Bool', 'char', 'short', 'int', 'long', 'signed', 'unsigned', 'float', 'double',
              'auto', '__int64'}
ATTR_WORDS = {'__attribute__', '__launch_bounds__', 'alignas', '__align__', '__declspec', '__builtin_align__'}
ASSIGN_OPS = {'=', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^='}
BIN_PREC = [('||',), ('&&',), ('|',), ('^',), ('&',), ('==', '!='), ('<', '>', '<=', '>='), ('<<', '>>'),
            ('+', '-'), ('*', '/', '%')]
CXX_CASTS = {'static_cast', 'reinterpret_cast', 'const_cast', 'dynamic_cast'}
TEMPLATE_FUNCS = {'tiled_partition', 'thread_block_tile', 'plus', 'less', 'greater', 'bit_and', 'bit_or',
                  'bit_xor', 'min', 'max', '__pk_vec'}


class Parser:
    def __init__(self, toks, env, ns=()):
        self.toks = []
        self.pragmas = {}
        pend = []
        for k, t in toks:
            if k in ('ws', 'nl'):
                continue
            if k == 'pp':
                s = t.lstrip('#').strip()
                if s.startswith('pragma'):
                    pend.append(s[6:].strip())
                continue
            if pend:
                self.pragmas[len(self.toks)] = pend
                pend = []
            self.toks.append((k, t))
        self.toks.append(('eof', ''))
        self.i = 0
        self.env = env
        self.ns = list(ns)

    # -- token helpers ------------------------------------------------------
    def peek(self, o=0):
        j = self.i + o
        return self.toks[j][1] if j < len(self.toks) else ''

    def kind(self, o=0):
        j = self.i + o
        return self.toks[j][0] if j < len(self.toks) else 'eof'

    def next(self):
        t = self.toks[self.i][1]
        self.i += 1
        return t

    def accept(self, t):
        if self.peek() == t:
            self.i += 1
            return True
        return False

    def expect(self, t):
        if self.peek() != t:
            raise ParseError('expected %r, found %r near %s' % (t, self.peek(), self.context()))
        self.i += 1

    def context(self):
        return ' '.join(x[1] for x in self.toks[max(0, self.i - 6):self.i + 6])

    def skip_balanced(self, open_t, close_t):
        depth = 0
        while True:
            t = self.next()
            if t == open_t:
                depth += 1
            elif t == close_t:
                depth -= 1
                if depth == 0:
                    return
            elif t == '':
                raise ParseError('unbalanced %s' % open_t)

    def ident(self):
        if self.kind() != 'id':
            raise ParseError('expected a name, found %r near %s' % (self.peek(), self.context()))
        return self.next()

    def qualified_name(self):
        """name, ns::name, ::name (no template arguments)."""
        parts = []
        if self.peek() == '::':
            self.next()
        parts.append(self.ident())
        while self.peek() == '::' and self.kind(1) == 'id':
            self.next()
            parts.append(self.next())
        return '::'.join(parts)

    # -- types ---------------------------------------------------------------
    def lookup_type(self, name):
        return self.env.lookup_type(name, self.ns)

    def peek_qualified(self):
        """Return (name, ntoks) of a qualified name at the cursor without consuming it."""
        j = self.i
        if self.toks[j][1] == '::':
            j += 1
        if self.toks[j][0] != 'id':
            return None, 0
        parts = [self.toks[j][1]]
        j += 1
        while self.toks[j][1] == '::' and self.toks[j + 1][0] == 'id':
            parts.append(self.toks[j + 1][1])
            j += 2
        return '::'.join(parts), j - self.i

    def is_type_start(self, o=0):
        t = self.peek(o)
        if t in BASE_WORDS or t in ('struct', 'union', 'enum', 'class', 'const', 'volatile', 'unsigned', 'signed'):
            return True
        if t in SPECIFIERS or t in ATTR_WORDS:
            return True
        if self.kind(o) == 'id' or t == '::':
            save = self.i
            self.i += o
            name, n = self.peek_qualified()
            self.i = save
            if name is None:
                return False
            if self.lookup_type(name) is not None:
                return True
            if name.split('::')[-1] == 'thread_block_tile' or name.split('::')[-1] in ('plus', 'less', 'greater',
                                                                                    'bit_and', 'bit_or', 'bit_xor'):
                return self.peek(o + n) == '<'
        return False

    def parse_attrs(self, storage):
        while True:
            t = self.peek()
            if t in ('__attribute__', '__declspec'):
                self.next()
                self.skip_balanced('(', ')')
            elif t in ('__launch_bounds__',):
                self.next()
                start = self.i
                self.skip_balanced('(', ')')
                storage.add(('launch_bounds', tuple(x[1] for x in self.toks[start + 1:self.i - 1])))
            elif t in ('alignas', '__align__', '__builtin_align__'):
                self.next()
                self.expect('(')
                e = self.expr()
                self.expect(')')
                storage.add(('align', self.env.const_eval(e)))
            elif t == '[' and self.peek(1) == '[':
                self.skip_balanced('[', ']')
            else:
                return

    def decl_specs(self):
        """Returns (base type, storage set). Defines struct/enum types it meets."""
        storage = set()
        words = []
        base = None
        while True:
            self.parse_attrs(storage)
            t = self.peek()
            if t in SPECIFIERS:
                storage.add(self.next())
                continue
            if t in BASE_WORDS:
                words.append(self.next())
                continue
            if t in ('struct', 'union', 'class') and base is None and not words:
                base = self.struct_spec()
                continue
            if t == 'enum' and base is None and not words:
                base = self.enum_spec()
                continue
            if base is None and not words and (self.kind() == 'id' or t == '::'):
                name, n = self.peek_qualified()
                last = name.split('::')[-1] if name else ''
                if last == 'thread_block_tile' and self.peek(n) == '<':
                    self.i += n
                    targs = self.template_args()
                    base = TGroup('tile', self.env.const_eval(targs[0]))
                    continue
                if last in ('plus', 'less', 'greater', 'bit_and', 'bit_or', 'bit_xor') and self.peek(n) == '<':
                    self.i += n
                    targs = self.template_args()
                    base = TFunctor(last, targs[0])
                    continue
                ty = self.lookup_type(name) if name else None
                if ty is not None:
                    self.i += n
                    base = ty
                    continue
            break
        if base is None:
            base = self.base_from_words(words)
        elif words:
            raise ParseError('bad type: %s %r' % (' '.join(words), base))
        return base, storage

    def base_from_words(self, words):
        if not words:
            raise ParseError('expected a type near %s' % self.context())
        if words == ['auto']:
            return None   # deduced from the initialiser
        w = [x for x in words if x not in ('signed',)]
        signed = 'signed' in words
        unsigned = 'unsigned' in w
        w = [x for x in w if x != 'unsigned']
        if w == ['void']:
            return VOID
        if w in (['bool'], ['_Bool']):
            return BOOL
        if w == ['char']:
            return UCHAR if unsigned else (SCHAR if signed else CHAR)
        if w in (['short'], ['short', 'int'], ['int', 'short']):
            return USHORT if unsigned else SHORT
        if w in ([], ['int']):
            return UINT if unsigned else INT
        if w in (['long'], ['long', 'int'], ['int', 'long']):
            return ULONG if unsigned else LONG
        if sorted(w) in (['long', 'long'], ['int', 'long', 'long']) or w == ['__int64']:
            return ULLONG if unsigned else LLONG
        if w == ['float']:
            return FLOAT
        if w in (['double'], ['long', 'double']):
            return DOUBLE
        raise ParseError('unsupported type %s' % ' '.join(words))

    def struct_spec(self):
        kw = self.next()
        storage = set()
        self.parse_attrs(storage)
        name = None
        if self.kind() == 'id':
            name = self.qualified_name()
        self.parse_attrs(storage)
        if self.peek() != '{' and self.peek() != ':':
            t = self.env.lookup_struct(name, self.ns) if name else None
            if t is None:
                t = TStruct(name, kw == 'union')
                if name:
                    self.env.define_struct(name, t, self.ns)
            return t
        if self.peek() == ':':      # base classes: not supported
            raise ParseError('struct with base class')
        t = self.env.lookup_struct(name, self.ns, local=True) if name else None
        if t is None or t.complete:
            t = TStruct(name, kw == 'union')
        if name:
            self.env.define_struct(name, t, self.ns)
        self.expect('{')
        members = []
        min_align = 1
        for k, v in storage:
            if k == 'align':
                min_align = v
        while not self.accept('}'):
            if self.peek() in ('public', 'private', 'protected') and self.peek(1) == ':':
                self.i += 2
                continue
            if self.peek() == ';':
                self.next()
                continue
            if self.peek() in ('template', 'using', 'typedef', 'static_assert', 'friend'):
                self.skip_member()
                t.methods = True
                continue
            mbase, mst = self.decl_specs()
            if self.peek() == ';' and isinstance(mbase, TStruct) and mbase.name is None:   # anonymous union/struct
                self.next()
                members.append((None, mbase))
                continue
            # constructor / destructor / method
            if self.peek() == '(' or (self.peek() == '~') or self.is_method_ahead():
                self.skip_member()
                t.methods = True
                continue
            while True:
                name2, mk = self.declarator()
                mt = mk(mbase)
                if self.peek() == '{' or (isinstance(mt, TFunc)):
                    # member function definition or declaration
                    self.skip_rest_of_member()
                    t.methods = True
                    break
                if self.peek() == ':':     # bit-field
                    raise ParseError('bit-fields are not supported')
                if 'static' in mst:
                    pass
                else:
                    members.append((name2, mt))
                if self.accept('='):
                    self.assign()
                if not self.accept(','):
                    break
            if self.peek() != '}':
                self.accept(';')
        t.layout(members, min_align)
        return t

    def is_method_ahead(self):
        # e.g. "operator+(" or a constructor named after the struct
        return self.peek() == 'operator'

    def skip_member(self):
        depth = 0
        while True:
            t = self.next()
            if t in ('(', '['):
                depth += 1
            elif t in (')', ']'):
                depth -= 1
            elif t == '{' and depth == 0:
                self.i -= 1
                self.skip_balanced('{', '}')
                self.accept(';')
                return
            elif t == ';' and depth == 0:
                return
            elif t == '':
                return

    def skip_rest_of_member(self):
        while self.peek() not in ('{', ';', ''):
            if self.peek() == '(':
                self.skip_balanced('(', ')')
            else:
                self.next()
        if self.peek() == '{':
            self.skip_balanced('{', '}')
            self.accept(';')
        else:
            self.accept(';')

    def enum_spec(self):
        self.next()
        scoped = self.accept('class') or self.accept('struct')
        name = None
        if self.kind() == 'id':
            name = self.qualified_name()
        under = INT
        if self.accept(':'):
            under, _ = self.decl_specs()
        if self.peek() != '{':
            return under
        self.expect('{')
        val = -1
        while not self.accept('}'):
            cname = self.ident()
            if self.accept('='):
                val = self.env.const_eval(self.assign())
            else:
                val += 1
            self.env.define_enum_const(cname, val, under, self.ns, (name if scoped else None))
            if name and not scoped:
                self.env.define_enum_const(name + '::' + cname, val, under, self.ns, None)
            if not self.accept(','):
                self.expect('}')
                break
        if name:
            self.env.define_typedef(name, under, self.ns)
        return under

    def template_args(self):
        """After a name, at '<': parse template arguments (types or constant expressions)."""
        self.expect('<')
        args = []
        while True:
            if self.peek() == '>':
                self.next()
                break
            if self.peek() == '>>':
                self.toks[self.i] = ('punct', '>')
                break
            if self.is_type_start():
                base, _ = self.decl_specs()
                _, mk = self.declarator(abstract=True)
                args.append(mk(base))
            else:
                args.append(self.expr_no_gt())
            if self.accept(','):
                continue
            if self.peek() == '>>':
                self.toks[self.i] = ('punct', '>')
                self.toks.insert(self.i + 1, ('punct', '>'))
            self.expect('>')
            break
        return args

    def expr_no_gt(self):
        """A template argument expression: stops at '>' (no relational/shift operators)."""
        return self.binary(len(BIN_PREC) - 3)

    def declarator(self, abstract=False):
        """Returns (name, make(base) -> type)."""
        ptrs = []
        while True:
            t = self.peek()
            if t == '*':
                self.next()
                restrict = False
                while self.peek() in ('const', 'volatile', '__restrict__', '__restrict', 'restrict'):
                    if self.next() != 'const' and self.toks[self.i - 1][1] != 'volatile':
                        restrict = True
                ptrs.append(('*', restrict))
            elif t in ('&', '&&'):
                self.next()
                ptrs.append(('&', False))
            elif t in ATTR_WORDS or t in ('__restrict__', '__restrict', 'restrict', 'const', 'volatile'):
                if t in ATTR_WORDS:
                    st = set()
                    self.parse_attrs(st)
                else:
                    self.next()
            else:
                break
        inner = None
        name = None
        if self.peek() == '(' and (self.peek(1) in ('*', '&') or (self.kind(1) == 'id' and self.peek(2) == ')'
                                                                  and not self.is_type_start(1))):
            self.next()
            name, inner = self.declarator(abstract)
            self.expect(')')
        elif self.kind() == 'id' and self.peek() not in SPECIFIERS:
            name = self.qualified_name()
        elif not abstract and self.peek() == 'operator':
            raise ParseError('operator overloading is not supported')
        suffixes = []
        while True:
            st = set()
            self.parse_attrs(st)
            if self.peek() == '[':
                self.next()
                if self.accept(']'):
                    suffixes.append(('[', None))
                else:
                    n = self.expr()
                    self.expect(']')
                    suffixes.append(('[', n))
            elif self.peek() == '(':
                self.next()
                params, variadic = self.param_list()
                while self.peek() in ('const', 'noexcept', 'override', 'final', 'volatile'):
                    self.next()
                suffixes.append(('(', (params, variadic)))
            else:
                break

        env = self.env

        def make(base):
            t = base
            for kind, r in ptrs:
                t = TPtr(t, r) if kind == '*' else TRef(t)
            for kind, v in reversed(suffixes):
                if kind == '[':
                    n = None if v is None else env.const_eval(v)
                    t = TArr(t, n)
                else:
                    params, variadic = v
                    t = TFunc(t, params, variadic)
            if inner is not None:
                t = inner(t)
            return t
        make.suffixes = suffixes
        return name, make

    def param_list(self):
        params = []
        variadic = False
        if self.accept(')'):
            return params, variadic
        if self.peek() == 'void' and self.peek(1) == ')':
            self.i += 2
            return params, variadic
        while True:
            if self.accept('...'):
                variadic = True
                self.expect(')')
                break
            base, st = self.decl_specs()
            name, mk = self.declarator(abstract=True)
            t = mk(base)
            if isinstance(t, TArr):
                t = TPtr(t.elem)
            d = VarDecl(name, t, None, st)
            if self.accept('='):
                d.init = self.assign()
            params.append(d)
            if self.accept(','):
                continue
            self.expect(')')
            break
        return params, variadic

    def type_name(self):
        base, st = self.decl_specs()
        _, mk = self.declarator(abstract=True)
        return mk(base)

    # -- top level -----------------------------------------------------------
    def external(self):
        """Parse one top-level item. Returns a list of FuncDef / VarDecl / None."""
        out = []
        if self.peek() == 'typedef':
            self.next()
            base, st = self.decl_specs()
            while True:
                name, mk = self.declarator()
                if name:
                    self.env.define_typedef(name, mk(base), self.ns)
                if not self.accept(','):
                    break
            self.accept(';')
            return out
        if self.peek() == 'using':
            self.next()
            name = self.ident()
            self.expect('=')
            t = self.type_name()
            self.env.define_typedef(name, t, self.ns)
            self.accept(';')
            return out
        if self.peek() == 'template':
            raise ParseError('templates are not supported by the PTX compiler yet')
        base, storage = self.decl_specs()
        if self.peek() == ';':
            self.next()
            return out
        while True:
            name, mk = self.declarator()
            if base is None:
                raise ParseError("'auto' at file scope")
            t = mk(base)
            if isinstance(t, TFunc):
                kind = 'kernel' if '__global__' in storage else ('device' if '__device__' in storage else 'host')
                if self.peek() == '{':
                    body = self.block()
                    out.append(FuncDef(name, self.ns, t.ret, t.params, body, kind, storage))
                    return out
                out.append(FuncDef(name, self.ns, t.ret, t.params, None, kind, storage))
            else:
                d = VarDecl(name, t, None, storage)
                d.ns = self.ns
                if self.accept('='):
                    d.init = self.initializer()
                elif self.peek() == '{':
                    d.init = self.initializer()
                out.append(d)
            if not self.accept(','):
                break
        self.accept(';')
        return out

    # -- statements ------------------------------------------------------------
    def block(self):
        self.expect('{')
        stmts = []
        self.env.push_scope()
        try:
            while not self.accept('}'):
                if self.peek() == '':
                    raise ParseError('unexpected end of function body')
                stmts.append(self.statement())
        finally:
            self.env.pop_scope()
        return SBlock(stmts)

    def statement(self):
        pragmas = self.pragmas.get(self.i, [])
        t = self.peek()
        if t == '{':
            return self.block()
        if t == ';':
            self.next()
            return SEmpty()
        if t == 'if':
            self.next()
            self.accept('constexpr')
            self.expect('(')
            c = self.cond_expr()
            self.expect(')')
            th = self.statement()
            el = self.statement() if self.accept('else') else None
            return SIf(c, th, el)
        if t == 'while':
            self.next()
            self.expect('(')
            c = self.cond_expr()
            self.expect(')')
            return SWhile(c, self.statement())
        if t == 'do':
            self.next()
            body = self.statement()
            self.expect('while')
            self.expect('(')
            c = self.expr()
            self.expect(')')
            self.accept(';')
            return SDo(body, c)
        if t == 'for':
            self.next()
            self.expect('(')
            self.env.push_scope()
            try:
                if self.accept(';'):
                    init = None
                elif self.is_decl_start():
                    init = self.declaration()
                else:
                    init = SExpr(self.expr())
                    self.expect(';')
                c = None if self.peek() == ';' else self.expr()
                self.expect(';')
                step = None if self.peek() == ')' else self.expr()
                self.expect(')')
                body = self.statement()
            finally:
                self.env.pop_scope()
            unroll = None
            for p in pragmas:
                m = re.match(r'^unroll\b\s*\(?\s*([0-9]*)', p)
                if m:
                    unroll = int(m.group(1)) if m.group(1) else 0
                elif p.startswith('nounroll'):
                    unroll = 1
            return SFor(init, c, step, body, unroll)
        if t == 'switch':
            self.next()
            self.expect('(')
            e = self.expr()
            self.expect(')')
            return SSwitch(e, self.statement())
        if t == 'case':
            self.next()
            v = self.cond()
            self.expect(':')
            return SCase(v)
        if t == 'default' and self.peek(1) == ':':
            self.i += 2
            return SDefault()
        if t == 'break':
            self.next()
            self.accept(';')
            return SBreak()
        if t == 'continue':
            self.next()
            self.accept(';')
            return SContinue()
        if t == 'return':
            self.next()
            e = None if self.peek() == ';' else self.expr()
            self.accept(';')
            return SReturn(e)
        if t == 'goto' or (self.kind() == 'id' and self.peek(1) == ':' and self.peek(2) != ':'):
            raise ParseError('goto/labels are not supported')
        if t in ('asm', '__asm__', '__asm'):
            e = self.asm_stmt()
            return SExpr(e)
        if t in ('static_assert', 'typedef', 'using'):
            if t == 'typedef' or t == 'using':
                self.external()
                return SEmpty()
            self.skip_member()
            return SEmpty()
        if self.is_decl_start():
            return self.declaration()
        e = self.expr()
        self.expect(';')
        return SExpr(e)

    def cond_expr(self):
        if self.is_decl_start():
            raise ParseError('declarations in conditions are not supported')
        return self.expr()

    def asm_stmt(self):
        self.next()
        while self.peek() in ('volatile', '__volatile__', 'goto'):
            self.next()
        self.expect('(')
        text = b''
        while self.kind() == 'str':
            text += self.str_literal()
        outs, ins = [], []
        for lst in (outs, ins):
            if not self.accept(':'):
                break
            while self.kind() == 'str':
                cons = self.str_literal().decode()
                self.expect('(')
                e = self.expr()
                self.expect(')')
                lst.append((cons, e))
                if not self.accept(','):
                    break
        while self.accept(':'):
            while self.kind() == 'str':
                self.str_literal()
                self.accept(',')
        self.expect(')')
        self.accept(';')
        return EAsm(text.decode('utf-8', 'replace'), outs, ins)

    def is_decl_start(self):
        t = self.peek()
        if t in BASE_WORDS:
            # "float(x)" at statement start is an expression (functional cast)
            return not (self.peek(1) == '(' and t in ('float', 'double', 'int', 'bool', 'char', 'long', 'short'))
        if t in ('struct', 'union', 'enum', 'const', 'volatile', 'unsigned', 'signed', 'static', 'extern',
                 'constexpr', '__shared__', 'register', 'auto') or t in SPECIFIERS or t in ATTR_WORDS:
            return True
        if self.kind() == 'id' or t == '::':
            name, n = self.peek_qualified()
            if name is None:
                return False
            last = name.split('::')[-1]
            if last == 'thread_block_tile' and self.peek(n) == '<':
                return True
            ty = self.lookup_type(name)
            if ty is None:
                return False
            nt = self.peek(n)
            return nt not in ('(', '{', '::') or self.kind(n + 1) == 'id' and nt == '('
        return False

    def declaration(self):
        base, storage = self.decl_specs()
        decls = []
        if self.peek() == ';':
            self.next()
            return SDecl(decls)
        while True:
            name, mk = self.declarator()
            t = mk(base) if base is not None else None
            d = VarDecl(name, t, None, storage)
            if self.accept('='):
                d.init = self.initializer()
            elif self.peek() == '{':
                d.init = self.initializer()
            elif self.peek() == '(':
                self.next()
                args = []
                if not self.accept(')'):
                    while True:
                        args.append(self.assign())
                        if self.accept(')'):
                            break
                        self.expect(',')
                d.init = args[0] if len(args) == 1 else EInit(args)
            if t is None and d.init is None:
                raise ParseError("'auto' without an initialiser")
            if isinstance(t, TArr) and t.n is None and isinstance(d.init, EInit):
                t = TArr(t.elem, len(d.init.items))
                d.t = t
            elif isinstance(t, TArr) and t.n is None and isinstance(d.init, EStr):
                t = TArr(t.elem, len(d.init.v) + 1)
                d.t = t
            if name:
                self.env.declare_local(name)
            decls.append(d)
            if not self.accept(','):
                break
        self.expect(';')
        return SDecl(decls)

    def initializer(self):
        if self.peek() == '{':
            self.next()
            items = []
            while not self.accept('}'):
                if self.peek() == '.' and self.kind(1) == 'id':   # designated initialiser
                    self.next()
                    fname = self.next()
                    self.expect('=')
                    items.append(('.', fname, self.initializer()))
                else:
                    items.append(self.initializer())
                if not self.accept(','):
                    self.expect('}')
                    break
            return EInit(items)
        return self.assign()

    # -- expressions -------------------------------------------------------------
    def expr(self):
        e = self.assign()
        while self.peek() == ',':
            self.next()
            e = EComma(e, self.assign())
        return e

    def assign(self):
        lhs = self.cond()
        t = self.peek()
        if t in ASSIGN_OPS:
            self.next()
            rhs = self.initializer() if self.peek() == '{' else self.assign()
            return EAssign(t, lhs, rhs)
        return lhs

    def cond(self):
        c = self.binary(0)
        if self.accept('?'):
            a = self.expr()
            self.expect(':')
            b = self.assign()
            return ECond(c, a, b)
        return c

    def binary(self, level):
        if level >= len(BIN_PREC):
            return self.cast()
        e = self.binary(level + 1)
        ops = BIN_PREC[level]
        while self.peek() in ops:
            op = self.next()
            rhs = self.binary(level + 1)
            e = EBin(op, e, rhs)
        return e

    def cast(self):
        if self.peek() == '(' and self.is_type_start(1):
            save = self.i
            self.next()
            try:
                t = self.type_name()
                self.expect(')')
            except ParseError:
                self.i = save
                return self.unary()
            if self.peek() == '{':     # compound literal
                return ECast(t, self.initializer(), 'c')
            return ECast(t, self.cast(), 'c')
        return self.unary()

    def unary(self):
        t = self.peek()
        if t in ('++', '--'):
            self.next()
            return EUnary(t + 'pre', self.unary())
        if t in ('+', '-', '!', '~', '*', '&'):
            self.next()
            return EUnary(t, self.cast())
        if t == 'sizeof':
            self.next()
            if self.peek() == '(' and self.is_type_start(1):
                self.next()
                ty = self.type_name()
                self.expect(')')
                return ESizeof(ty, None)
            return ESizeof(None, self.unary())
        if t in ('alignof', '__alignof__', '_Alignof'):
            self.next()
            self.expect('(')
            ty = self.type_name()
            self.expect(')')
            return ENum(ty.align, ULONG)
        return self.postfix(self.primary())

    def postfix(self, e):
        while True:
            t = self.peek()
            if t == '[':
                self.next()
                idx = self.expr()
                self.expect(']')
                e = EIndex(e, idx)
            elif t == '(':
                self.next()
                args = []
                if not self.accept(')'):
                    while True:
                        args.append(self.initializer() if self.peek() == '{' else self.assign())
                        if self.accept(')'):
                            break
                        self.expect(',')
                e = ECall(e, args)
            elif t in ('.', '->'):
                self.next()
                if self.peek() == 'template':
                    self.next()
                name = self.ident()
                e = EMember(e, name, t == '->')
                if self.peek() == '<' and name in TEMPLATE_FUNCS:
                    e = EMember(e.obj, name, e.arrow)
                    targs = self.template_args()
                    e.name = (name, tuple(targs))
            elif t in ('++', '--'):
                self.next()
                e = EUnary(t + 'post', e)
            else:
                return e

    def str_literal(self):
        k, t = self.toks[self.i]
        self.i += 1
        q = t.index('"')
        return unescape(t[q + 1:-1])

    def primary(self):
        k = self.kind()
        t = self.peek()
        if k == 'num':
            self.next()
            v, ty = parse_number(t)
            return ENum(v, ty)
        if k == 'chr':
            self.next()
            q = t.index("'")
            b = unescape(t[q + 1:-1])
            v = b[0] if b else 0
            return ENum(v - 256 if v > 127 else v, CHAR)
        if k == 'str':
            s = b''
            while self.kind() == 'str':
                s += self.str_literal()
            return EStr(s)
        if t == '(':
            self.next()
            e = self.expr()
            self.expect(')')
            return e
        if t in ('true', 'false'):
            self.next()
            return ENum(1 if t == 'true' else 0, BOOL)
        if t in ('nullptr', 'NULL'):
            self.next()
            return ECast(TPtr(VOID), ENum(0, LONG), 'c')
        if t in CXX_CASTS:
            self.next()
            targs = self.template_args()
            self.expect('(')
            e = self.expr()
            self.expect(')')
            return ECast(targs[0], e, t)
        if t == '{':
            return self.initializer()
        if t in ('asm', '__asm__'):
            return self.asm_stmt()
        if t in BASE_WORDS or t in ('unsigned', 'signed'):
            ty = self.base_from_words(self.base_words())
            return self.functional_cast(ty)
        if k == 'id' or t == '::':
            name, n = self.peek_qualified()
            ty = self.lookup_type(name) if name and not self.env.is_local(name) else None
            last = name.split('::')[-1] if name else ''
            if ty is not None and self.peek(n) in ('(', '{'):
                self.i += n
                return self.functional_cast(ty)
            if last in ('plus', 'less', 'greater', 'bit_and', 'bit_or', 'bit_xor') and self.peek(n) == '<':
                self.i += n
                targs = self.template_args()
                ty = TFunctor(last, targs[0])
                if self.accept('('):
                    self.expect(')')
                return EType(ty)
            self.i += n
            targs = None
            if self.peek() == '<' and last in TEMPLATE_FUNCS:
                targs = tuple(self.template_args())
            return EName(name, targs)
        raise ParseError('unexpected %r near %s' % (t, self.context()))

    def base_words(self):
        words = []
        while self.peek() in BASE_WORDS or self.peek() in ('unsigned', 'signed'):
            words.append(self.next())
        return words

    def functional_cast(self, ty):
        if self.peek() == '{':
            init = self.initializer()
            return ECast(ty, init, 'func')
        self.expect('(')
        args = []
        if not self.accept(')'):
            while True:
                args.append(self.assign())
                if self.accept(')'):
                    break
                self.expect(',')
        if len(args) == 1:
            return ECast(ty, args[0], 'func')
        return ECast(ty, EInit(args), 'func')


def f32_bits(v):
    return struct.unpack('<I', struct.pack('<f', v))[0]


def f64_bits(v):
    return struct.unpack('<Q', struct.pack('<d', v))[0]


def round_f32(v):
    try:
        return struct.unpack('<f', struct.pack('<f', v))[0]
    except OverflowError:
        return float('inf') if v > 0 else float('-inf')
