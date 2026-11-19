"""Tokenizer for preprocessed C++ (the output of `c++ -E`).

Tokens are mutable two-element lists [kind, text]; joining all texts gives back
the original source exactly. Kinds: ws, nl, pp (a whole preprocessor line such
as a line marker or #pragma), str, chr, num, id, punct.
"""
import re

_PUNCT = ['...', '<<=', '>>=', '->*', '::', '->', '++', '--', '<<', '>>', '<=', '>=', '==', '!=',
          '&&', '||', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '.*', '##']
_punct_re = '|'.join(re.escape(p) for p in _PUNCT) + r'|[{}\[\]()<>;:,.?~!%^&*+\-=/|#\\@`$]'
TOKEN_RE = re.compile(r'''
  (?P<ws>[ \t\r\f\v]+|\\\r?\n)
| (?P<nl>\n)
| (?P<str>(?:u8|u|U|L)?"(?:[^"\\\n]|\\.)*")
| (?P<chr>(?:u8|u|U|L)?'(?:[^'\\\n]|\\.)*')
| (?P<num>\.?[0-9](?:[eEpP][+-]|[A-Za-z0-9_.]|'(?=[0-9A-Za-z]))*)
| (?P<id>[A-Za-z_][A-Za-z0-9_]*)
| (?P<punct>''' + _punct_re + r''')
''', re.X | re.S)
RAW_RE = re.compile(r'(?:u8|u|U|L)?R"([^()\\ \t\n]{0,16})\(')

SKIP = ('ws', 'nl')


def tokenize(src):
    toks = []
    i, n = 0, len(src)
    bol = True
    while i < n:
        if bol:
            j = i
            while j < n and src[j] in ' \t':
                j += 1
            if j < n and src[j] == '#':
                k = j
                while True:
                    e = src.find('\n', k)
                    if e == -1:
                        e = n
                        break
                    if e > 0 and src[e - 1] == '\\':
                        k = e + 1
                        continue
                    break
                if j > i:
                    toks.append(['ws', src[i:j]])
                toks.append(['pp', src[j:e]])
                i = e
                bol = False
                continue
        c = src[i]
        if c in 'uULR':
            m = RAW_RE.match(src, i)
            if m:
                delim = m.group(1)
                end = src.find(')' + delim + '"', m.end())
                end = n if end == -1 else end + len(delim) + 2
                toks.append(['str', src[i:end]])
                i = end
                bol = False
                continue
        m = TOKEN_RE.match(src, i)
        if not m:
            toks.append(['punct', c])
            i += 1
            bol = False
            continue
        kind = m.lastgroup
        toks.append([kind, m.group()])
        i = m.end()
        bol = kind == 'nl'
    return toks


def significant(toks, keep_pp=False):
    """Indices of tokens that are not whitespace (and not pp lines unless keep_pp)."""
    if keep_pp:
        return [i for i, t in enumerate(toks) if t[0] not in SKIP]
    return [i for i, t in enumerate(toks) if t[0] not in SKIP and t[0] != 'pp']


def match_forward(toks, sig, i, open_t, close_t):
    """sig[i] is open_t; return index j (into sig) of the matching close_t."""
    depth = 0
    pairs = {'(': ')', '[': ']', '{': '}'}
    for j in range(i, len(sig)):
        t = toks[sig[j]][1]
        if t == open_t:
            depth += 1
        elif t == close_t:
            depth -= 1
            if depth == 0:
                return j
    return -1
