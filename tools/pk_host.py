"""Host pass: turns preprocessed CUDA C++ into plain C++ for the host compiler.

* removes __global__/__device__/__host__/__constant__/__managed__
* __shared__            -> static thread_local   (one copy per block on the CPU)
* extern __shared__ T a[] -> T* a = (T*)::pk::cpu::dyn_smem()
* k<<<g, b, s, st>>>(args) -> ::pk::launch_k(k, "k", &::pk_tu_info, ::pk::LaunchCfg(g, b, s, st), args)
* appends the translation-unit registration record (OpenCL source, kernel
  signatures, symbols, launched kernels)
"""
from pk_lex import tokenize, significant

REMOVE = {'__global__', '__device__', '__host__', '__constant__', '__managed__', '__grid_constant__'}
ATTR_FUNCS = {'__attribute__', '__launch_bounds__', 'alignas', '__declspec', '__align__'}


class TranslateError(Exception):
    pass


def _brace_kind(toks, sig, idx):
    """Classify the '{' at sig[idx]: ('ns', name|None) | ('class', None) | ('other', None)."""
    j = idx - 1
    collected = []
    while j >= 0:
        t = toks[sig[j]][1]
        if t in (';', '{', '}'):
            break
        collected.append(t)
        j -= 1
    collected.reverse()
    if 'namespace' in collected:
        k = collected.index('namespace')
        name = collected[k + 1] if k + 1 < len(collected) and collected[k + 1] != '{' else '(anon)'
        rest = collected[k + 1:]
        if len(rest) > 1 and '::' in rest:  # namespace a::b {
            name = ''.join(rest)
        return ('ns', name)
    if collected[:1] == ['extern'] and len(collected) == 2 and toks[sig[idx - 1]][0] == 'str':
        return ('ns', None)
    depth = 0
    has_paren = False
    for t in collected:
        if t == '(':
            depth += 1
            has_paren = True
        elif t == ')':
            depth -= 1
    if not has_paren and any(t in ('struct', 'class', 'union', 'enum') for t in collected) and '=' not in collected:
        return ('class', None)
    return ('other', None)


def _skip_attr(toks, sig, j):
    """If sig[j] starts an attribute construct, return the index after it, else j."""
    t = toks[sig[j]][1]
    if t in ATTR_FUNCS and j + 1 < len(sig) and toks[sig[j + 1]][1] == '(':
        depth = 0
        k = j + 1
        while k < len(sig):
            tt = toks[sig[k]][1]
            if tt == '(':
                depth += 1
            elif tt == ')':
                depth -= 1
                if depth == 0:
                    return k + 1
            k += 1
        return k
    if t == '[' and j + 1 < len(sig) and toks[sig[j + 1]][1] == '[':
        k = j + 2
        while k + 1 < len(sig) and not (toks[sig[k]][1] == ']' and toks[sig[k + 1]][1] == ']'):
            k += 1
        return k + 2
    return j


def translate_host(src, dev, file_label):
    """src: host-pass preprocessed text. dev: dict from the device pass. Returns (text, info)."""
    toks = tokenize(src)
    sig = significant(toks)
    n = len(sig)
    T = lambda j: toks[sig[j]][1] if 0 <= j < n else ''

    kernels = _discover_kernels(toks, sig)
    replacements = []     # (first_tok, last_tok, text)

    # ---- pass B: qualifiers and __shared__ (edit token texts in place) ----
    edits = {}
    stack = []
    for j in range(n):
        t = T(j)
        if t == '{':
            stack.append(_brace_kind(toks, sig, j))
        elif t == '}':
            if stack:
                stack.pop()
        elif t in REMOVE:
            edits[sig[j]] = ''
        elif t == '__noinline__':
            if T(j - 1) != '(':
                edits[sig[j]] = '__attribute__((noinline))'
        elif t == '__shared__':
            b = j - 1
            while b >= 0 and T(b) not in (';', '{', '}', ':'):
                b -= 1
            e = j + 1
            while e < n and T(e) != ';':
                e += 1
            stmt = list(range(b + 1, e))
            if any(T(k) == 'extern' for k in stmt):
                if not any(k[0] == 'other' for k in stack):
                    raise TranslateError('extern __shared__ at namespace scope is not supported')
                parts = []
                k = b + 1
                while k < e:
                    k2 = _skip_attr(toks, sig, k)
                    if k2 != k:
                        k = k2
                        continue
                    if T(k) not in ('extern', '__shared__') and T(k) not in REMOVE:
                        parts.append(k)
                    k += 1
                br = [k for k in parts if T(k) == '[']
                if not br:
                    raise TranslateError('cannot parse extern __shared__ declaration')
                name_idx = parts[parts.index(br[0]) - 1]
                type_txt = ' '.join(T(k) for k in parts if k < name_idx)
                text = '%s* %s = reinterpret_cast<%s*>(::pk::cpu::dyn_smem())' % (type_txt, T(name_idx), type_txt)
                nl = sum(toks[x][1].count('\n') for x in range(sig[b + 1], sig[e - 1] + 1))
                replacements.append((sig[b + 1], sig[e - 1], text + '\n' * nl))
            else:
                edits[sig[j]] = 'static thread_local'
    for k, v in edits.items():
        toks[k][1] = v

    # ---- pass C: kernel launches ----
    launched = []
    generic = False
    dyn = False
    i = 0
    while i < n:
        if T(i) == '<<' and i + 1 < n and T(i + 1) == '<' and sig[i + 1] == sig[i] + 1:
            r = _launch(toks, sig, i, kernels, file_label)
            replacements.append(r[0])
            if r[1]:
                launched.append(r[1])
            if r[2] == 'dyn':
                dyn = True
            elif r[2]:
                generic = True
            i = r[3]
            continue
        i += 1
    out = _apply(toks, replacements)

    # kernel entries for cudaLaunchKernel: non-template, defined here, unique name
    entries = []
    for name, kinfo in kernels.items():
        if kinfo['defined'] and not kinfo['template'] and kinfo['count'] == 1 and not kinfo['in_class']:
            ref = '::' + '::'.join(kinfo['nspath'] + [name])
            entries.append((ref, name))
    info = {'launched': launched, 'generic': generic, 'dyn': dyn, 'entries': entries, 'kernels': kernels,
            'has_templates': any(k['template'] for k in kernels.values())}
    out += registration(dev, info, file_label)
    return out, info


def _discover_kernels(toks, sig):
    n = len(sig)
    T = lambda j: toks[sig[j]][1] if 0 <= j < n else ''
    kernels = {}
    stack = []
    for i in range(n):
        t = T(i)
        if t == '{':
            stack.append(_brace_kind(toks, sig, i))
        elif t == '}':
            if stack:
                stack.pop()
        elif t == '__global__':
            j = i + 1
            while j < n:
                k2 = _skip_attr(toks, sig, j)
                if k2 != j:
                    j = k2
                    continue
                if T(j) == '(':
                    break
                if T(j) in (';', '{', '='):
                    j = -1
                    break
                j += 1
            if not (0 < j < n and toks[sig[j - 1]][0] == 'id'):
                continue
            name = T(j - 1)
            b = i - 1
            is_tmpl = False
            while b >= 0 and T(b) not in (';', '{', '}'):
                if T(b) == 'template':
                    is_tmpl = True
                b -= 1
            depth, e = 0, j
            while e < n:
                if T(e) == '(':
                    depth += 1
                elif T(e) == ')':
                    depth -= 1
                    if depth == 0:
                        break
                e += 1
            f = e + 1
            while f < n and T(f) not in ('{', ';'):
                f += 1
            defined = T(f) == '{'
            # parameter default values (needed when a launch omits trailing arguments)
            params, cur, d = [], [], 0
            for q in range(j + 1, e):
                tt = T(q)
                if tt in ('(', '[', '{', '<'):
                    d += 1
                elif tt in (')', ']', '}', '>'):
                    d -= 1
                if tt == ',' and d == 0:
                    params.append(cur)
                    cur = []
                else:
                    cur.append(q)
            if cur:
                params.append(cur)
            defaults = []
            for prm in params:
                eq = [q for q in prm if T(q) == '=']
                defaults.append(''.join(toks[x][1] for x in range(sig[eq[0] + 1], sig[prm[-1]] + 1)).strip()
                                if eq else None)
            if defaults == [None] and len(params) == 1 and [T(q) for q in params[0]] in (['void'], []):
                defaults = []
            nsp = [k[1] for k in stack if k[0] == 'ns' and k[1] not in (None, '(anon)')]
            info = kernels.setdefault(name, {'template': False, 'defined': False, 'count': 0, 'nspath': nsp,
                                             'in_class': any(k[0] == 'class' for k in stack), 'defaults': defaults})
            if any(x is not None for x in defaults):
                info['defaults'] = defaults
            info['template'] = info['template'] or is_tmpl
            if defined:
                info['count'] += 1
                info['defined'] = True
                info['nspath'] = nsp
    return kernels


def _launch(toks, sig, i, kernels, file_label):
    """Rewrite the launch whose '<<' '<' starts at sig[i]."""
    n = len(sig)
    T = lambda j: toks[sig[j]][1] if 0 <= j < n else ''
    # --- kernel expression (backwards) ---
    j = i - 1
    has_targs = False
    if T(j) in ('>', '>>'):
        depth = 0
        while j >= 0:
            t = T(j)
            if t == '>':
                depth += 1
            elif t == '>>':
                depth += 2
            elif t == '<':
                depth -= 1
            elif t == '<<':
                depth -= 2
            elif t in (';', '{', '}'):
                raise TranslateError('cannot parse kernel template arguments before <<<')
            if depth <= 0 and t in ('<', '<<'):
                break
            j -= 1
        j -= 1
        has_targs = True
    start = j
    simple = False
    if T(j) == ')':
        depth = 0
        while j >= 0:
            if T(j) == ')':
                depth += 1
            elif T(j) == '(':
                depth -= 1
                if depth == 0:
                    break
            j -= 1
        start = j
    elif T(j) == ']':
        depth = 0
        while j >= 0:
            if T(j) == ']':
                depth += 1
            elif T(j) == '[':
                depth -= 1
                if depth == 0:
                    break
            j -= 1
        start = j - 1
    elif toks[sig[j]][0] == 'id':
        simple = True
        start = j
        while T(start - 1) == '::':
            if toks[sig[start - 2]][0] == 'id':
                start -= 2
            else:
                start -= 1
                break
        while T(start - 1) in ('.', '->') and toks[sig[start - 2]][0] == 'id':
            simple = False
            start -= 2
    else:
        raise TranslateError('cannot find the kernel name before <<<')
    base = T(i - 1) if not has_targs else None
    if has_targs:
        # identifier just before the '<' that opens the template args
        k = i - 1
        depth = 0
        while k >= 0:
            t = T(k)
            if t == '>':
                depth += 1
            elif t == '>>':
                depth += 2
            elif t == '<':
                depth -= 1
            elif t == '<<':
                depth -= 2
            if depth <= 0 and t in ('<', '<<'):
                break
            k -= 1
        base = T(k - 1)
    # --- launch configuration ---
    j = i + 2
    depth = 0
    cfg_start = j
    while j < n:
        t = T(j)
        if t in ('(', '[', '{'):
            depth += 1
        elif t in (')', ']', '}'):
            depth -= 1
        elif depth == 0 and t == '>>' and T(j + 1) == '>' and sig[j + 1] == sig[j] + 1:
            break
        j += 1
    if j >= n:
        raise TranslateError('unterminated <<< ... >>> launch configuration')
    cfg_end = j - 1
    p = j + 2
    if T(p) != '(':
        raise TranslateError('expected ( after >>>')
    depth = 0
    q = p
    while q < n:
        t = T(q)
        if t in ('(', '[', '{'):
            depth += 1
        elif t in (')', ']', '}'):
            depth -= 1
            if depth == 0:
                break
        q += 1
    text_of = lambda a, b: ''.join(toks[x][1] for x in range(sig[a], sig[b] + 1)) if b >= a else ''
    expr = text_of(start, i - 1).strip()
    cfg = text_of(cfg_start, cfg_end).strip()
    args = text_of(p + 1, q - 1).strip()
    info = kernels.get(base)
    if info and info.get('defaults'):
        nargs = 0
        if args:
            nargs, d = 1, 0
            for q in range(p + 1, q):
                tt = T(q)
                if tt in ('(', '[', '{'):
                    d += 1
                elif tt in (')', ']', '}'):
                    d -= 1
                elif tt == ',' and d == 0:
                    nargs += 1
            q = q + 1
        dfl = info['defaults']
        if nargs < len(dfl) and all(x is not None for x in dfl[nargs:]):
            args = ', '.join(([args] if args else []) + dfl[nargs:])
    use_k = (simple or has_targs) and (has_targs or (info is not None and not info['template']))
    deduced_template = info is not None and info['template'] and not has_targs
    if use_k:
        text = '::pk::launch_k(%s, "%s", &::pk_tu_info, ::pk::LaunchCfg(%s)%s)' % (
            expr, base, cfg, (', ' + args) if args else '')
        name = base if not has_targs else None
        is_generic = has_targs
    elif not deduced_template:
        # function pointer (variable, array element, member): resolved by address at run time
        text = '::pk::launch_dyn(%s, &::pk_tu_info, ::pk::LaunchCfg(%s)%s)' % (
            expr, cfg, (', ' + args) if args else '')
        name = None
        is_generic = 'dyn'
    else:
        text = ('::pk::launch_generic(&::pk_tu_info, ::pk::LaunchCfg(%s), [&](auto... pk_a_) { %s(pk_a_...); }%s)'
                % (cfg, expr, (', ' + args) if args else ''))
        name = None
        is_generic = True
    orig_nl = sum(toks[x][1].count('\n') for x in range(sig[start], sig[q] + 1))
    text += '\n' * max(0, orig_nl - text.count('\n'))
    return (sig[start], sig[q], text), name, is_generic, q + 1


def _apply(toks, replacements):
    replacements.sort(key=lambda r: r[0])
    out = []
    pos = 0
    for a, b, text in replacements:
        if a < pos:
            raise TranslateError('overlapping rewrites (nested launch?)')
        out.extend(t[1] for t in toks[pos:a])
        out.append(text)
        pos = b + 1
    out.extend(t[1] for t in toks[pos:])
    return ''.join(out)


def _cstr(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'


def registration(dev, info, file_label):
    lines = ['', '// ---- generated by pkcc: translation-unit registration ----', 'namespace {']
    cl_src = dev.get('cl_src') if dev.get('cl_ok') else None
    if cl_src:
        delim = 'PKCL'
        while (')' + delim + '"') in cl_src:
            delim += 'X'
        lines.append('static const char pk_cl_src__[] = R"%s(%s)%s";' % (delim, cl_src, delim))
    kern = dev.get('kernels', []) if cl_src else []
    if kern:
        lines.append('static const ::pk::KernelMeta pk_cl_kernels__[] = {%s};' % ', '.join(
            '{%s, %s, %s}' % (_cstr(k[0]), _cstr(k[1]), _cstr(','.join(k[2]) if len(k) > 2 else '')) for k in kern))
    if info['launched']:
        lines.append('static const char* const pk_launched__[] = {%s};' % ', '.join(
            _cstr(x) for x in sorted(set(info['launched']))))
    ptx_src = dev.get('ptx_src') if dev.get('ptx_ok') else None
    if ptx_src:
        delim = 'PKPTX'
        while (')' + delim + '"') in ptx_src:
            delim += 'X'
        lines.append('static const char pk_ptx_src__[] = R"%s(%s)%s";' % (delim, ptx_src, delim))
        pk = dev.get('ptx_kernels', [])
        if pk:
            lines.append('static const char* const pk_ptx_kernels__[] = {%s};' % ', '.join(_cstr(k) for k in pk))
    syms = list(dev.get('symbols', [])) if cl_src else []
    if ptx_src:
        have = set(nm for _, nm in syms)
        for ref, nm in dev.get('ptx_symbols', []):
            if nm not in have:
                syms.append((ref, nm))
                have.add(nm)
    if syms:
        lines.append('static const ::pk::SymbolMeta pk_symbols__[] = {%s};' % ', '.join(
            '{(const void*)&%s, sizeof(%s), %s}' % (ref, ref, _cstr(nm)) for ref, nm in syms))
    if info['entries']:
        lines.append('static const ::pk::KernelEntry pk_entries__[] = {%s};' % ', '.join(
            '::pk::make_kernel_entry(&%s, %s)' % (ref, _cstr(nm)) for ref, nm in info['entries']))
    body = [
        '::pk::TuInfo& t = ::pk_tu_info;',
        't.file = %s;' % _cstr(file_label),
        't.needs_fibers = %s;' % ('true' if dev.get('needs_fibers', True) else 'false'),
        't.cl_ok = %s;' % ('true' if dev.get('cl_ok') else 'false'),
        't.cl_reason = %s;' % _cstr(dev.get('reason') or ''),
    ]
    if cl_src:
        body.append('t.cl_src = pk_cl_src__;')
    body.append('t.ptx_ok = %s;' % ('true' if dev.get('ptx_ok') else 'false'))
    body.append('t.ptx_reason = %s;' % _cstr(dev.get('ptx_reason') or ''))
    if ptx_src:
        body.append('t.ptx_src = pk_ptx_src__;')
        if dev.get('ptx_kernels'):
            body.append('t.ptx_kernels = pk_ptx_kernels__; t.nptx_kernels = %d;' % len(dev['ptx_kernels']))
    if kern:
        body.append('t.kernels = pk_cl_kernels__; t.nkernels = %d;' % len(kern))
    if info['launched']:
        body.append('t.launched = pk_launched__; t.nlaunched = %d;' % len(set(info['launched'])))
    body.append('t.has_generic_launch = %s;' % ('true' if info['generic'] else 'false'))
    body.append('t.has_dyn_launch = %s;' % ('true' if info.get('dyn') else 'false'))
    body.append('t.has_template_kernels = %s;' % ('true' if info.get('has_templates') else 'false'))
    if syms:
        body.append('t.symbols = pk_symbols__; t.nsymbols = %d;' % len(syms))
    if info['entries']:
        body.append('t.entries = pk_entries__; t.nentries = %d;' % len(info['entries']))
    body.append('::pk::register_tu(&t);')
    lines.append('struct PkTuInit__ { PkTuInit__() { %s } } pk_tu_init__;' % ' '.join(body))
    lines.append('}  // namespace')
    return '\n'.join(lines) + '\n'
