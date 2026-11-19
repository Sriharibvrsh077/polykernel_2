// ptxsim: a stand-in for the NVIDIA driver library (libcuda) that interprets PTX on the CPU.
// It implements the driver calls PolyKernel's PTX back end uses and the PTX that
// tools/pk_ptx.py emits, so the PTX path can be tested on machines without an NVIDIA GPU:
//
//   c++ -std=c++17 -O2 -shared -fPIC -frounding-math -ffp-contract=off -o libptxsim.so ptxsim.cpp
//   PK_CUDA_DRIVER_LIBRARY=/path/to/libptxsim.so PK_BACKEND=ptx ./program
//
// Semantics follow NVIDIA's public PTX ISA document: state spaces, integer/float
// arithmetic and conversions with their rounding modes, CTA barriers (bar.sync/bar.red),
// warp-synchronous shfl/vote/match/activemask with member masks, atomics, vprintf, malloc.
// Blocks run one after another; the 32 lanes of each warp run in lock step (one
// instruction per lane per round), which is how pre-Volta GPUs execute convergent code.
#include <algorithm>
#include <atomic>
#include <thread>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

typedef int CUresult;
typedef unsigned long long CUdeviceptr;

struct Err : std::runtime_error {
  explicit Err(const std::string& s) : std::runtime_error(s) {}
};

// ---------------------------------------------------------------------------------------
// Program representation
// ---------------------------------------------------------------------------------------
enum Kind { K_NONE, K_REG, K_IMM, K_SPECIAL, K_MEMREG, K_MEMSYM, K_SYM, K_VEC };
enum Special { SP_TID, SP_NTID, SP_CTAID, SP_NCTAID, SP_LANEID, SP_LMLT, SP_LMLE, SP_LMGT, SP_LMGE, SP_LMEQ,
               SP_CLOCK, SP_CLOCK64, SP_GTIMER, SP_WARPID, SP_NWARPID, SP_SMID, SP_NSMID, SP_GRIDID,
               SP_DYNSMEM };
enum Space { S_GENERIC, S_GLOBAL, S_SHARED, S_LOCAL, S_CONST, S_PARAM };

struct Opnd {
  Kind k = K_NONE;
  int reg = -1, reg2 = -1;   // reg2: second destination of "d|p"
  int vr[4] = {-1, -1, -1, -1};
  int nv = 0;                // K_VEC: number of registers in {a, b, ...}
  bool neg = false;          // !%p
  int sym = -1;
  int sp = 0, comp = 0;
  int64_t off = 0;
  uint64_t imm = 0;
};

struct TI {   // type information of a PTX type suffix
  int sz = 0;
  bool sg = false, fl = false, pr = false, ok = false;
};

TI type_info(const std::string& t) {
  TI r;
  if (t == "pred") {
    r.sz = 1;
    r.pr = r.ok = true;
    return r;
  }
  if (t.size() < 2 || !strchr("busf", t[0])) return r;
  for (size_t i = 1; i < t.size(); ++i)
    if (!isdigit(static_cast<unsigned char>(t[i]))) return r;
  int bits = atoi(t.c_str() + 1);
  if (bits != 8 && bits != 16 && bits != 32 && bits != 64) return r;
  r.sz = bits / 8;
  r.sg = t[0] == 's';
  r.fl = t[0] == 'f';
  r.ok = true;
  return r;
}

enum Op {
  O_LD, O_ST, O_MOV, O_CVT, O_CVTA, O_ADD, O_SUB, O_MUL, O_DIV, O_REM, O_MIN, O_MAX, O_AND, O_OR, O_XOR,
  O_SHL, O_SHR, O_MUL24, O_MAD, O_MAD24, O_FMA, O_NEG, O_ABS, O_NOT, O_CNOT, O_SQRT, O_RSQRT, O_RCP, O_EX2,
  O_LG2, O_SIN, O_COS, O_TANH, O_POPC, O_CLZ, O_BREV, O_BFIND, O_BFE, O_BFI, O_SETP, O_SELP, O_TESTP,
  O_PRMT, O_SHF, O_FNS, O_ATOM, O_RED, O_NOP, O_ISSPACEP, O_CALL, O_RET, O_TRAP, O_BRA, O_BAR, O_BARWARP,
  O_SHFL, O_VOTE, O_MATCH, O_ACTIVEMASK, O_COPYSIGN, O_SAD
};

enum Cmp { C_EQ, C_NE, C_LT, C_LE, C_GT, C_GE, C_LO, C_LS, C_HI, C_HS, C_EQU, C_NEU, C_LTU, C_LEU, C_GTU, C_GEU,
           C_NUM, C_NAN };
enum AOp { A_ADD, A_EXCH, A_MIN, A_MAX, A_INC, A_DEC, A_AND, A_OR, A_XOR, A_CAS };
enum Mode { M_UP, M_DOWN, M_BFLY, M_IDX, M_ALL, M_ANY, M_UNI, M_BALLOT, M_POPC, M_AND, M_OR, M_NAN, M_NUM,
            M_FINITE, M_INF, M_NORMAL, M_SUBNORMAL };

struct Ins {
  std::string op;   // full opcode text, for messages
  std::vector<std::string> mods;
  Op code = O_NOP;
  Opnd o[6];
  int n = 0;
  int pred = -1;
  bool predneg = false;
  int target = -1;
  TI t, t2;                 // type (cvt: destination) and cvt source type
  int rm = FE_TONEAREST;    // float rounding mode
  bool rint = false;        // cvt .rni/.rzi/.rmi/.rpi
  bool hi = false, wide = false, sat = false, to = false, sync = false, shiftamt = false;
  Space sp = S_GENERIC;
  int sub = 0;              // Cmp / AOp / Mode
  std::string callee;
  std::vector<int> callargs;
  int callret = -1;
};

struct Sym {
  std::string name;
  Space space = S_GLOBAL;
  size_t size = 0, align = 1;
  uint64_t addr = 0;   // global/const: host address; shared/local/param: offset
  bool kparam = false, cparam = false, dyn = false;
};

struct Entry {
  std::string name;
  std::vector<int> params;
  size_t param_bytes = 0;
  std::vector<Ins> code;
  int nregs = 0;
  size_t local_bytes = 0;
  std::vector<uint64_t> shoff;   // offset of each static .shared variable this kernel uses
  size_t shared_static = 0;
};

struct Module {
  std::vector<Sym> syms;
  std::map<std::string, int> symidx;
  std::vector<std::unique_ptr<Entry>> entries;
  std::vector<std::unique_ptr<std::vector<uint8_t>>> storage;
  size_t cparam_bytes = 0;
};

std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  std::string cur;
  int depth = 0;
  for (char ch : s) {
    if (ch == '[' || ch == '{' || ch == '(') depth++;
    if (ch == ']' || ch == '}' || ch == ')') depth--;
    if (ch == c && depth == 0) {
      out.push_back(trim(cur));
      cur.clear();
    } else {
      cur += ch;
    }
  }
  if (!trim(cur).empty()) out.push_back(trim(cur));
  return out;
}

std::vector<std::string> words(const std::string& s) {
  std::vector<std::string> out;
  std::string w;
  for (char c : s + " ") {
    if (c == ' ' || c == '\t') {
      if (!w.empty()) out.push_back(w);
      w.clear();
    } else {
      w += c;
    }
  }
  return out;
}

uint64_t parse_int(const std::string& s0) {
  std::string s = trim(s0);
  bool neg = false;
  if (!s.empty() && s[0] == '-') {
    neg = true;
    s = s.substr(1);
  }
  uint64_t v;
  if (s.size() > 2 && s[0] == '0' && (s[1] == 'f' || s[1] == 'F' || s[1] == 'd' || s[1] == 'D'))
    v = std::stoull(s.substr(2), nullptr, 16);
  else if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
    v = std::stoull(s.substr(2), nullptr, 16);
  else
    v = std::stoull(s);
  return neg ? static_cast<uint64_t>(-static_cast<int64_t>(v)) : v;
}

const std::map<std::string, Op>& op_table() {
  static const std::map<std::string, Op> t = {
      {"ld", O_LD}, {"ldu", O_LD}, {"st", O_ST}, {"mov", O_MOV}, {"cvt", O_CVT}, {"cvta", O_CVTA},
      {"add", O_ADD}, {"sub", O_SUB}, {"mul", O_MUL}, {"div", O_DIV}, {"rem", O_REM}, {"min", O_MIN},
      {"max", O_MAX}, {"and", O_AND}, {"or", O_OR}, {"xor", O_XOR}, {"shl", O_SHL}, {"shr", O_SHR},
      {"mul24", O_MUL24}, {"mad", O_MAD}, {"mad24", O_MAD24}, {"fma", O_FMA}, {"neg", O_NEG}, {"abs", O_ABS},
      {"not", O_NOT}, {"cnot", O_CNOT}, {"sqrt", O_SQRT}, {"rsqrt", O_RSQRT}, {"rcp", O_RCP}, {"ex2", O_EX2},
      {"lg2", O_LG2}, {"sin", O_SIN}, {"cos", O_COS}, {"tanh", O_TANH}, {"popc", O_POPC}, {"clz", O_CLZ},
      {"brev", O_BREV}, {"bfind", O_BFIND}, {"bfe", O_BFE}, {"bfi", O_BFI}, {"setp", O_SETP},
      {"selp", O_SELP}, {"testp", O_TESTP}, {"prmt", O_PRMT}, {"shf", O_SHF}, {"fns", O_FNS},
      {"atom", O_ATOM}, {"red", O_RED}, {"membar", O_NOP}, {"fence", O_NOP}, {"nanosleep", O_NOP},
      {"prefetch", O_NOP}, {"prefetchu", O_NOP}, {"pmevent", O_NOP}, {"isspacep", O_ISSPACEP},
      {"call", O_CALL}, {"ret", O_RET}, {"exit", O_RET}, {"trap", O_TRAP}, {"brkpt", O_TRAP}, {"bra", O_BRA},
      {"bar", O_BAR}, {"barrier", O_BAR}, {"shfl", O_SHFL}, {"vote", O_VOTE}, {"match", O_MATCH},
      {"activemask", O_ACTIVEMASK}, {"copysign", O_COPYSIGN}, {"sad", O_SAD}};
  return t;
}

void decode(Ins& in) {
  auto it = op_table().find(in.mods[0]);
  if (it == op_table().end()) throw Err("unsupported instruction '" + in.op + "'");
  in.code = it->second;
  if ((in.code == O_BAR) && in.mods.size() > 1 && in.mods[1] == "warp") in.code = O_BARWARP;
  size_t nm = in.mods.size();
  in.t = type_info(in.mods.back());
  if (in.code == O_CVT && nm >= 3) {
    in.t = type_info(in.mods[nm - 2]);
    in.t2 = type_info(in.mods[nm - 1]);
  }
  for (size_t i = 1; i < nm; ++i) {
    const std::string& m = in.mods[i];
    if (m == "rz") in.rm = FE_TOWARDZERO;
    else if (m == "rp") in.rm = FE_UPWARD;
    else if (m == "rm") in.rm = FE_DOWNWARD;
    else if (m == "rni") in.rint = true;
    else if (m == "rzi") in.rint = true, in.rm = FE_TOWARDZERO;
    else if (m == "rpi") in.rint = true, in.rm = FE_UPWARD;
    else if (m == "rmi") in.rint = true, in.rm = FE_DOWNWARD;
    else if (m == "hi") in.hi = true;
    else if (m == "wide") in.wide = true;
    else if (m == "sat") in.sat = true;
    else if (m == "to") in.to = true;
    else if (m == "sync") in.sync = true;
    else if (m == "shiftamt") in.shiftamt = true;
    else if (m == "global") in.sp = S_GLOBAL;
    else if (m == "shared") in.sp = S_SHARED;
    else if (m == "local") in.sp = S_LOCAL;
    else if (m == "const") in.sp = S_CONST;
    else if (m == "param") in.sp = S_PARAM;
  }
  static const std::map<std::string, int> cmp = {
      {"eq", C_EQ}, {"ne", C_NE}, {"lt", C_LT}, {"le", C_LE}, {"gt", C_GT}, {"ge", C_GE}, {"lo", C_LO},
      {"ls", C_LS}, {"hi", C_HI}, {"hs", C_HS}, {"equ", C_EQU}, {"neu", C_NEU}, {"ltu", C_LTU},
      {"leu", C_LEU}, {"gtu", C_GTU}, {"geu", C_GEU}, {"num", C_NUM}, {"nan", C_NAN}};
  static const std::map<std::string, int> aop = {{"add", A_ADD}, {"exch", A_EXCH}, {"min", A_MIN},
                                                 {"max", A_MAX}, {"inc", A_INC}, {"dec", A_DEC},
                                                 {"and", A_AND}, {"or", A_OR}, {"xor", A_XOR}, {"cas", A_CAS}};
  static const std::map<std::string, int> mode = {
      {"up", M_UP}, {"down", M_DOWN}, {"bfly", M_BFLY}, {"idx", M_IDX}, {"all", M_ALL}, {"any", M_ANY},
      {"uni", M_UNI}, {"ballot", M_BALLOT}, {"popc", M_POPC}, {"and", M_AND}, {"or", M_OR},
      {"notanumber", M_NAN}, {"number", M_NUM}, {"finite", M_FINITE}, {"infinite", M_INF},
      {"normal", M_NORMAL}, {"subnormal", M_SUBNORMAL}};
  auto find_in = [&](const std::map<std::string, int>& tab, const char* what) {
    for (size_t i = 1; i < nm; ++i) {
      auto f = tab.find(in.mods[i]);
      if (f != tab.end()) return f->second;
    }
    throw Err(std::string("no ") + what + " in '" + in.op + "'");
  };
  switch (in.code) {
    case O_SETP: in.sub = find_in(cmp, "comparison"); break;
    case O_ATOM:
    case O_RED: in.sub = find_in(aop, "atomic operation"); break;
    case O_SHFL:
    case O_VOTE:
    case O_TESTP: in.sub = find_in(mode, "mode"); break;
    case O_MATCH: in.sub = in.mods[1] == "all" ? M_ALL : M_ANY; break;
    case O_BAR:
      in.sub = -1;
      if (nm > 1 && in.mods[1] == "red") in.sub = find_in(mode, "reduction");
      break;
    default: break;
  }
  bool typed = in.code != O_BAR && in.code != O_BARWARP && in.code != O_RET && in.code != O_TRAP &&
               in.code != O_BRA && in.code != O_NOP && in.code != O_ISSPACEP && in.code != O_CALL;
  if (typed && !in.t.ok) throw Err("instruction '" + in.op + "' has no operand type");
  if (in.code == O_CVT && !in.t2.ok) throw Err("conversion '" + in.op + "' has no source type");
}

struct Parser {
  Module& m;
  Entry* e = nullptr;
  std::map<std::string, int> regs;
  std::map<std::string, int> labels;
  std::vector<std::pair<int, std::string>> fixups;
  explicit Parser(Module& mm) : m(mm) {}

  int add_sym(const Sym& s) {
    auto it = m.symidx.find(s.name);
    if (it != m.symidx.end()) return it->second;
    m.syms.push_back(s);
    m.symidx[s.name] = static_cast<int>(m.syms.size() - 1);
    return static_cast<int>(m.syms.size() - 1);
  }

  int reg(const std::string& name) {
    auto it = regs.find(trim(name));
    if (it == regs.end()) throw Err("unknown register " + name);
    return it->second;
  }

  Opnd operand(std::string s) {
    s = trim(s);
    Opnd o;
    if (s.empty()) return o;
    if (s[0] == '!') {
      o = operand(s.substr(1));
      o.neg = true;
      return o;
    }
    if (s[0] == '[') {
      std::string in = trim(s.substr(1, s.size() - 2));
      int64_t off = 0;
      size_t plus = in.find_first_of("+-", 1);
      std::string base = in;
      if (plus != std::string::npos) {
        base = trim(in.substr(0, plus));
        std::string o2 = trim(in.substr(plus + 1));
        off = static_cast<int64_t>(parse_int(o2));
        if (in[plus] == '-') off = -off;
      }
      if (base[0] == '%') {
        o.k = K_MEMREG;
        o.reg = reg(base);
      } else if (isdigit(static_cast<unsigned char>(base[0]))) {
        o.k = K_MEMREG;
        off += static_cast<int64_t>(parse_int(base));
      } else {
        o.k = K_MEMSYM;
        auto it = m.symidx.find(base);
        if (it == m.symidx.end()) throw Err("unknown symbol " + base);
        o.sym = it->second;
      }
      o.off = off;
      return o;
    }
    if (s[0] == '{') {
      auto parts = split(s.substr(1, s.size() - 2), ',');
      if (parts.size() > 4) throw Err("vector operand too long: " + s);
      o.k = K_VEC;
      o.nv = static_cast<int>(parts.size());
      for (size_t i = 0; i < parts.size(); ++i) o.vr[i] = parts[i] == "_" ? -1 : reg(parts[i]);
      return o;
    }
    size_t bar = s.find('|');
    if (bar != std::string::npos && s[0] == '%') {
      o.k = K_REG;
      o.reg = reg(s.substr(0, bar));
      o.reg2 = reg(s.substr(bar + 1));
      return o;
    }
    if (s[0] == '%') {
      static const std::map<std::string, Special> sp = {
          {"%tid", SP_TID}, {"%ntid", SP_NTID}, {"%ctaid", SP_CTAID}, {"%nctaid", SP_NCTAID},
          {"%laneid", SP_LANEID}, {"%lanemask_lt", SP_LMLT}, {"%lanemask_le", SP_LMLE},
          {"%lanemask_gt", SP_LMGT}, {"%lanemask_ge", SP_LMGE}, {"%lanemask_eq", SP_LMEQ},
          {"%clock", SP_CLOCK}, {"%clock64", SP_CLOCK64}, {"%globaltimer", SP_GTIMER},
          {"%warpid", SP_WARPID}, {"%nwarpid", SP_NWARPID}, {"%smid", SP_SMID}, {"%nsmid", SP_NSMID},
          {"%gridid", SP_GRIDID}, {"%dynamic_smem_size", SP_DYNSMEM}};
      size_t dot = s.find('.');
      std::string base = dot == std::string::npos ? s : s.substr(0, dot);
      auto it = sp.find(base);
      if (it != sp.end()) {
        o.k = K_SPECIAL;
        o.sp = it->second;
        if (dot != std::string::npos) o.comp = s[dot + 1] - 'x';
        return o;
      }
      o.k = K_REG;
      o.reg = reg(s);
      return o;
    }
    if (isdigit(static_cast<unsigned char>(s[0])) || s[0] == '-') {
      o.k = K_IMM;
      o.imm = parse_int(s);
      return o;
    }
    auto it = m.symidx.find(s);
    if (it != m.symidx.end()) {
      o.k = K_SYM;
      o.sym = it->second;
      return o;
    }
    throw Err("bad operand '" + s + "'");
  }

  // [.extern|.visible|.weak] .global|.const|.shared|.local|.param [.align N] [.vN] .type name[[N]] [= init];
  void var_decl(const std::string& line, bool in_entry) {
    std::string l = trim(line);
    if (!l.empty() && l.back() == ';') l.pop_back();
    size_t eq = l.find('=');
    std::string decl = trim(eq == std::string::npos ? l : l.substr(0, eq));
    std::string init = eq == std::string::npos ? "" : trim(l.substr(eq + 1));
    Sym s;
    bool have_space = false, ext = false;
    TI ty;
    int vec = 1;
    std::string namedims;
    std::vector<std::string> w = words(decl);
    for (size_t i = 0; i < w.size(); ++i) {
      const std::string& x = w[i];
      if (x == ".extern") ext = true;
      else if (x == ".visible" || x == ".weak" || x == ".common") continue;
      else if (x == ".global") s.space = S_GLOBAL, have_space = true;
      else if (x == ".const") s.space = S_CONST, have_space = true;
      else if (x == ".shared") s.space = S_SHARED, have_space = true;
      else if (x == ".local") s.space = S_LOCAL, have_space = true;
      else if (x == ".param") s.space = S_PARAM, have_space = true;
      else if (x == ".align" && i + 1 < w.size()) s.align = std::stoul(w[++i]);
      else if (x.size() > 2 && x[0] == '.' && x[1] == 'v' && isdigit(static_cast<unsigned char>(x[2])))
        vec = atoi(x.c_str() + 2);
      else if (x[0] == '.' && type_info(x.substr(1)).ok) ty = type_info(x.substr(1));
      else if (x[0] == '.') continue;   // .ptr and other attributes
      else namedims = x;
    }
    if (!have_space || !ty.ok || namedims.empty()) throw Err("cannot parse declaration: " + line);
    size_t elsize = static_cast<size_t>(ty.sz) * vec;
    size_t br = namedims.find('[');
    s.name = namedims.substr(0, br);
    size_t count = 1;
    if (br != std::string::npos) {
      std::string n = namedims.substr(br + 1, namedims.find(']') - br - 1);
      if (n.empty()) {
        s.dyn = true;
        count = 0;
      } else {
        count = std::stoul(n);
      }
    }
    s.size = elsize * count;
    if (s.align < elsize && br == std::string::npos) s.align = elsize;
    if (s.align == 0) s.align = 1;
    if (s.space == S_LOCAL) {
      if (!in_entry) throw Err("module-scope .local is not supported");
      e->local_bytes = std::max(e->local_bytes, s.size);
      s.addr = 0;
      int id = add_sym(s);
      m.syms[id].size = std::max(m.syms[id].size, s.size);
      return;
    }
    if (s.space == S_SHARED) {
      if (ext || s.dyn) s.dyn = true;
      add_sym(s);
      return;
    }
    if (s.space == S_PARAM) {   // call parameter inside { } around a call
      if (m.symidx.count(s.name)) return;
      s.cparam = true;
      size_t sz = std::max<size_t>(8, (s.size + 7) / 8 * 8);
      m.cparam_bytes = (m.cparam_bytes + 7) / 8 * 8;
      s.addr = m.cparam_bytes;
      m.cparam_bytes += sz;
      add_sym(s);
      return;
    }
    // .global / .const data lives in host memory (ptxsim's "device memory")
    m.storage.emplace_back(new std::vector<uint8_t>(s.size ? s.size : 1, 0));
    std::vector<uint8_t>& buf = *m.storage.back();
    if (!init.empty()) {
      std::string body = init;
      if (body[0] == '{') body = body.substr(1, body.rfind('}') - 1);
      size_t pos = 0;
      for (const std::string& v : split(body, ',')) {
        uint64_t x = parse_int(v);
        if (pos + elsize <= buf.size()) memcpy(buf.data() + pos, &x, std::min<size_t>(elsize, 8));
        pos += elsize;
      }
    }
    s.addr = reinterpret_cast<uint64_t>(buf.data());
    add_sym(s);
  }

  void parse(const std::string& text) {
    std::vector<std::string> lines;
    std::string cur;
    for (char c : text) {
      if (c == '\n') {
        lines.push_back(cur);
        cur.clear();
      } else {
        cur += c;
      }
    }
    lines.push_back(cur);
    for (size_t li = 0; li < lines.size(); ++li) {
      std::string l = trim(lines[li]);
      size_t cm = l.find("//");
      if (cm != std::string::npos) l = trim(l.substr(0, cm));
      if (l.empty()) continue;
      if (l.rfind(".version", 0) == 0 || l.rfind(".target", 0) == 0 || l.rfind(".address_size", 0) == 0)
        continue;
      if (l.find(".func") != std::string::npos) {   // extern declarations (vprintf, malloc, free)
        if (l.find(".extern") == std::string::npos)
          throw Err("PTX functions (.func) are not supported; PolyKernel inlines device functions");
        while (l.find(';') == std::string::npos && li + 1 < lines.size()) l = trim(lines[++li]);
        continue;
      }
      if (l.find(".entry") != std::string::npos) {
        std::string head = l;
        while (head.find(')') == std::string::npos && li + 1 < lines.size()) head += " " + trim(lines[++li]);
        parse_entry(head, lines, li);
        continue;
      }
      if (l[0] == '.') {
        var_decl(l, false);
        continue;
      }
      throw Err("unexpected line: " + l);
    }
  }

  void parse_entry(const std::string& head, const std::vector<std::string>& lines, size_t& li) {
    m.entries.emplace_back(new Entry);
    e = m.entries.back().get();
    regs.clear();
    labels.clear();
    fixups.clear();
    size_t lp = head.find('(');
    size_t ep = head.find(".entry") + 6;
    e->name = trim(head.substr(ep, lp - ep));
    std::string plist = head.substr(lp + 1, head.rfind(')') - lp - 1);
    size_t off = 0;
    for (const std::string& p : split(plist, ',')) {
      if (p.empty()) continue;
      Sym s;
      s.space = S_PARAM;
      s.kparam = true;
      std::vector<std::string> w = words(p);
      TI ty;
      for (size_t i = 0; i < w.size(); ++i) {
        if (w[i] == ".align" && i + 1 < w.size()) s.align = std::stoul(w[++i]);
        else if (w[i][0] == '.' && type_info(w[i].substr(1)).ok) ty = type_info(w[i].substr(1));
      }
      std::string last = w.back();
      size_t br = last.find('[');
      s.name = last.substr(0, br);
      if (br != std::string::npos) {
        s.size = std::stoul(last.substr(br + 1)) * static_cast<size_t>(ty.ok ? ty.sz : 1);
      } else {
        s.size = ty.ok ? static_cast<size_t>(ty.sz) : 8;
        s.align = s.size;
      }
      if (s.align == 0) s.align = 1;
      off = (off + s.align - 1) / s.align * s.align;
      s.addr = off;
      off += s.size;
      e->params.push_back(add_sym(s));
    }
    e->param_bytes = off;
    while (++li < lines.size() && trim(lines[li]).rfind("{", 0) != 0) {
    }
    std::vector<std::string> body;
    int depth = 1;
    while (++li < lines.size()) {
      std::string l = trim(lines[li]);
      size_t cm = l.find("//");
      if (cm != std::string::npos) l = trim(l.substr(0, cm));
      if (l == "{") {
        depth++;
        continue;
      }
      if (l == "}") {
        if (--depth == 0) break;
        continue;
      }
      if (!l.empty()) body.push_back(l);
    }
    for (const std::string& l : body) {   // declarations first
      if (l.rfind(".reg", 0) == 0) {
        std::string d = trim(l.substr(4));
        if (!d.empty() && d.back() == ';') d.pop_back();
        std::vector<std::string> w = words(d);
        std::string rest;
        for (size_t i = 1; i < w.size(); ++i) rest += w[i];
        for (const std::string& r : split(rest, ',')) {
          size_t lt = r.find('<');
          if (lt == std::string::npos) {
            regs[r] = e->nregs++;
          } else {
            std::string pre = r.substr(0, lt);
            int n = std::stoi(r.substr(lt + 1));
            for (int i = 0; i < n; ++i) regs[pre + std::to_string(i)] = e->nregs++;
          }
        }
      } else if (l.rfind(".local", 0) == 0 || l.rfind(".param", 0) == 0 || l.rfind(".shared", 0) == 0) {
        var_decl(l, true);
      } else if (l[0] == '.' && l.rfind(".pragma", 0) != 0 && l.rfind(".maxntid", 0) != 0 &&
                 l.rfind(".minnctapersm", 0) != 0 && l.rfind(".reqntid", 0) != 0) {
        throw Err("unsupported directive in kernel body: " + l);
      }
    }
    for (const std::string& l : body) {
      if (l[0] == '.') continue;
      if (l.back() == ':') {
        labels[l.substr(0, l.size() - 1)] = static_cast<int>(e->code.size());
        continue;
      }
      instruction(l);
    }
    for (auto& f : fixups) {
      auto it = labels.find(f.second);
      if (it == labels.end()) throw Err("unknown label " + f.second);
      e->code[f.first].target = it->second;
    }
    // lay out the static shared variables this kernel references (as ptxas does per kernel)
    e->shoff.assign(m.syms.size(), ~0ULL);
    for (const Ins& in : e->code)
      for (int i = 0; i < in.n; ++i) {
        const Opnd& o = in.o[i];
        if ((o.k != K_SYM && o.k != K_MEMSYM) || o.sym < 0) continue;
        const Sym& y = m.syms[o.sym];
        if (y.space != S_SHARED || y.dyn || e->shoff[o.sym] != ~0ULL) continue;
        e->shared_static = (e->shared_static + y.align - 1) / y.align * y.align;
        e->shoff[o.sym] = e->shared_static;
        e->shared_static += y.size;
      }
    Ins ret;
    ret.op = "ret";
    ret.mods = {"ret"};
    ret.code = O_RET;
    e->code.push_back(ret);
  }

  void instruction(std::string l) {
    if (!l.empty() && l.back() == ';') l.pop_back();
    l = trim(l);
    Ins in;
    if (l[0] == '@') {
      size_t sp = l.find_first_of(" \t");
      std::string p = l.substr(1, sp - 1);
      if (p[0] == '!') {
        in.predneg = true;
        p = p.substr(1);
      }
      in.pred = reg(p);
      l = trim(l.substr(sp));
    }
    size_t sp = l.find_first_of(" \t");
    in.op = sp == std::string::npos ? l : l.substr(0, sp);
    std::string rest = sp == std::string::npos ? "" : trim(l.substr(sp));
    std::string cur;
    for (char c : in.op + ".") {
      if (c == '.') {
        in.mods.push_back(cur);
        cur.clear();
      } else {
        cur += c;
      }
    }
    decode(in);
    if (in.code == O_BRA) {
      fixups.push_back(std::make_pair(static_cast<int>(e->code.size()), rest));
      e->code.push_back(in);
      return;
    }
    if (in.code == O_CALL) {   // call.uni (ret), name, (a, b);   or   call.uni name, (a);
      std::string r = rest;
      if (r[0] == '(') {
        in.callret = m.symidx.at(trim(r.substr(1, r.find(')') - 1)));
        r = trim(r.substr(r.find(')') + 1));
        if (!r.empty() && r[0] == ',') r = trim(r.substr(1));
      }
      size_t c = r.find(',');
      in.callee = trim(r.substr(0, c));
      if (c != std::string::npos) {
        std::string args = r.substr(r.find('(', c) + 1);
        args = args.substr(0, args.find(')'));
        for (const std::string& a : split(args, ',')) in.callargs.push_back(m.symidx.at(a));
      }
      if (in.callee != "vprintf" && in.callee != "malloc" && in.callee != "free")
        throw Err("call to unsupported function " + in.callee);
      e->code.push_back(in);
      return;
    }
    auto ops = split(rest, ',');
    in.n = static_cast<int>(ops.size());
    if (in.n > 6) throw Err("too many operands: " + l);
    for (int i = 0; i < in.n; ++i) in.o[i] = operand(ops[i]);
    e->code.push_back(in);
  }
};

// ---------------------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------------------
enum Status { RUN, BAR, WARP, DONE };

struct Thread {
  std::vector<uint64_t> r;
  int pc = 0;
  Status st = RUN;
  unsigned tid[3] = {0, 0, 0};
  int lane = 0, warp = 0;
  uint64_t icount = 0;
  std::vector<uint8_t> local;
  std::vector<uint8_t> cparam;
};

struct Launch {
  Module* m;
  Entry* e;
  unsigned grid[3], block[3], ctaid[3];
  unsigned shmem = 0;
  std::vector<uint8_t> params;
  std::vector<uint8_t> shared;
  size_t dyn_off = 0;
};

std::mutex g_print_mu, g_atomic_mu;
std::atomic<uint64_t> g_clock{0};
const uint64_t kMaxInstructionsPerThread = 2000000000ULL;

float f32(uint64_t b) {
  uint32_t u = static_cast<uint32_t>(b);
  float f;
  memcpy(&f, &u, 4);
  return f;
}
double f64(uint64_t b) {
  double d;
  memcpy(&d, &b, 8);
  return d;
}
uint64_t bf32(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}
uint64_t bf64(double d) {
  uint64_t u;
  memcpy(&u, &d, 8);
  return u;
}
int64_t sext(uint64_t v, int bytes) {
  if (bytes >= 8) return static_cast<int64_t>(v);
  int sh = 64 - bytes * 8;
  return static_cast<int64_t>(v << sh) >> sh;
}
uint64_t zext(uint64_t v, int bytes) {
  if (bytes >= 8) return v;
  return v & ((1ULL << (bytes * 8)) - 1);
}
// A register value normalised to the type: sign-extended for signed types, zero-extended otherwise.
uint64_t norm(uint64_t v, const TI& t) { return t.sg ? static_cast<uint64_t>(sext(v, t.sz)) : zext(v, t.sz); }

struct RoundGuard {   // sets the host rounding mode for one PTX instruction
  int old;
  bool on;
  explicit RoundGuard(int rm) : old(0), on(rm != FE_TONEAREST) {
    if (on) {
      old = fegetround();
      fesetround(rm);
    }
  }
  ~RoundGuard() {
    if (on) fesetround(old);
  }
};

struct Exec {
  Launch& L;
  std::vector<Thread>& T;
  const std::vector<Ins>& code;
  Exec(Launch& l, std::vector<Thread>& t) : L(l), T(t), code(l.e->code) {}

  uint64_t special(const Thread& t, const Opnd& o) {
    switch (o.sp) {
      case SP_TID: return t.tid[o.comp];
      case SP_NTID: return L.block[o.comp];
      case SP_CTAID: return L.ctaid[o.comp];
      case SP_NCTAID: return L.grid[o.comp];
      case SP_LANEID: return static_cast<uint64_t>(t.lane);
      case SP_LMLT: return (1u << t.lane) - 1u;
      case SP_LMLE: return t.lane == 31 ? 0xffffffffu : (2u << t.lane) - 1u;
      case SP_LMGT: return t.lane == 31 ? 0u : (~((2u << t.lane) - 1u)) & 0xffffffffu;
      case SP_LMGE: return (~((1u << t.lane) - 1u)) & 0xffffffffu;
      case SP_LMEQ: return 1u << t.lane;
      case SP_CLOCK: return static_cast<uint32_t>(++g_clock);
      case SP_CLOCK64: return ++g_clock;
      case SP_GTIMER: return ++g_clock * 1000;
      case SP_WARPID: return static_cast<uint64_t>(t.warp);
      case SP_NWARPID: return (static_cast<uint64_t>(L.block[0]) * L.block[1] * L.block[2] + 31) / 32;
      case SP_SMID: return 0;
      case SP_NSMID: return 40;
      case SP_GRIDID: return 1;
      case SP_DYNSMEM: return L.shmem;
    }
    return 0;
  }

  uint64_t sym_addr(int s) {
    const Sym& y = L.m->syms[s];
    if (y.space == S_SHARED) return y.dyn ? L.dyn_off : L.e->shoff[s];
    return y.addr;
  }

  uint64_t val(const Thread& t, const Opnd& o) {
    switch (o.k) {
      case K_REG: return o.neg ? (t.r[o.reg] == 0) : t.r[o.reg];
      case K_IMM: return o.imm;
      case K_SPECIAL: return special(t, o);
      case K_SYM: return sym_addr(o.sym);
      default: throw Err("bad source operand");
    }
  }

  uint8_t* addr(Thread& t, const Opnd& o, Space sp, size_t n) {
    uint64_t a;
    if (o.k == K_MEMSYM) {
      const Sym& y = L.m->syms[o.sym];
      if (y.kparam) {
        if (static_cast<size_t>(y.addr + o.off) + n > L.params.size()) throw Err("parameter access out of range");
        return L.params.data() + y.addr + o.off;
      }
      if (y.cparam) return t.cparam.data() + y.addr + o.off;
      a = sym_addr(o.sym) + static_cast<uint64_t>(o.off);
      sp = (y.space == S_GLOBAL || y.space == S_CONST) ? S_GLOBAL : y.space;
    } else if (o.k == K_MEMREG) {
      a = (o.reg >= 0 ? t.r[o.reg] : 0) + static_cast<uint64_t>(o.off);
    } else {
      throw Err("bad memory operand");
    }
    if (sp != S_PARAM && n > 1 && (a & (n - 1)) != 0) {   // GPUs fault on misaligned accesses
      char buf[96];
      snprintf(buf, sizeof buf, "misaligned %zu-byte access at 0x%llx", n, static_cast<unsigned long long>(a));
      throw Err(buf);
    }
    switch (sp) {
      case S_SHARED:
        if (a + n > L.shared.size()) throw Err("shared memory access out of range");
        return L.shared.data() + a;
      case S_LOCAL:
        if (a + n > t.local.size()) throw Err("local memory access out of range");
        return t.local.data() + a;
      case S_PARAM:
        if (a + n > L.params.size()) throw Err("parameter access out of range");
        return L.params.data() + a;
      default:
        if (a < 4096) throw Err("null pointer access (address " + std::to_string(a) + ")");
        return reinterpret_cast<uint8_t*>(a);
    }
  }

  void set(Thread& t, const Opnd& o, uint64_t v) {
    if (o.k != K_REG) throw Err("bad destination operand");
    t.r[o.reg] = v;
  }

  uint64_t arith(const Ins& in, Op op, uint64_t a, uint64_t b) {
    const TI& ty = in.t;
    int n = ty.sz;
    if (ty.fl) {
      RoundGuard rg(in.rm);
      if (n == 4) {
        float x = f32(a), y = f32(b);
        volatile float r = 0;
        switch (op) {
          case O_ADD: r = x + y; break;
          case O_SUB: r = x - y; break;
          case O_MUL: r = x * y; break;
          case O_DIV: r = x / y; break;
          case O_MIN: r = std::isnan(x) ? y : (std::isnan(y) ? x : (x < y ? x : (y < x ? y : (std::signbit(x) ? x : y)))); break;
          case O_MAX: r = std::isnan(x) ? y : (std::isnan(y) ? x : (x > y ? x : (y > x ? y : (std::signbit(x) ? y : x)))); break;
          default: throw Err("float operation " + in.op);
        }
        float rv = r;
        if (in.sat) rv = std::isnan(rv) ? 0.f : (rv < 0.f ? 0.f : (rv > 1.f ? 1.f : rv));
        return bf32(rv);
      }
      double x = f64(a), y = f64(b);
      volatile double r = 0;
      switch (op) {
        case O_ADD: r = x + y; break;
        case O_SUB: r = x - y; break;
        case O_MUL: r = x * y; break;
        case O_DIV: r = x / y; break;
        case O_MIN: r = std::isnan(x) ? y : (std::isnan(y) ? x : (x < y ? x : (y < x ? y : (std::signbit(x) ? x : y)))); break;
        case O_MAX: r = std::isnan(x) ? y : (std::isnan(y) ? x : (x > y ? x : (y > x ? y : (std::signbit(x) ? y : x)))); break;
        default: throw Err("double operation " + in.op);
      }
      return bf64(r);
    }
    bool s = ty.sg;
    uint64_t ma = zext(a, n), mb = zext(b, n);
    int64_t sa = sext(a, n), sb = sext(b, n);
    uint64_t r = 0;
    switch (op) {
      case O_ADD:
        if (in.sat && s && n == 4) {
          int64_t v = sa + sb;
          return static_cast<uint64_t>(v > INT32_MAX ? INT32_MAX : (v < INT32_MIN ? INT32_MIN : v));
        }
        r = ma + mb;
        break;
      case O_SUB:
        if (in.sat && s && n == 4) {
          int64_t v = sa - sb;
          return static_cast<uint64_t>(v > INT32_MAX ? INT32_MAX : (v < INT32_MIN ? INT32_MIN : v));
        }
        r = ma - mb;
        break;
      case O_MUL:
        if (in.wide) {
          if (s) return static_cast<uint64_t>(sa * sb);
          return ma * mb;
        }
        if (in.hi) {
          if (n == 8) {
            if (s) return static_cast<uint64_t>(static_cast<int64_t>((static_cast<__int128>(sa) * sb) >> 64));
            return static_cast<uint64_t>((static_cast<unsigned __int128>(ma) * mb) >> 64);
          }
          if (s) return static_cast<uint64_t>((sa * sb) >> (8 * n));
          return (ma * mb) >> (8 * n);
        }
        r = ma * mb;
        break;
      case O_DIV:
        if (mb == 0) return s ? zext(~0ULL, n) : zext(~0ULL, n);
        if (s && sb == -1) r = static_cast<uint64_t>(-sa);
        else r = s ? static_cast<uint64_t>(sa / sb) : ma / mb;
        break;
      case O_REM:
        if (mb == 0) return ma;
        if (s && sb == -1) r = 0;
        else r = s ? static_cast<uint64_t>(sa % sb) : ma % mb;
        break;
      case O_MIN: r = s ? static_cast<uint64_t>(sa < sb ? sa : sb) : (ma < mb ? ma : mb); break;
      case O_MAX: r = s ? static_cast<uint64_t>(sa > sb ? sa : sb) : (ma > mb ? ma : mb); break;
      case O_AND: r = ma & mb; break;
      case O_OR: r = ma | mb; break;
      case O_XOR: r = ma ^ mb; break;
      case O_SHL: {
        uint64_t c = zext(b, 4);
        r = c >= static_cast<uint64_t>(n * 8) ? 0 : ma << c;
        break;
      }
      case O_SHR: {
        uint64_t c = zext(b, 4);
        if (s) r = static_cast<uint64_t>(c >= static_cast<uint64_t>(n * 8) ? (sa < 0 ? -1 : 0) : sa >> c);
        else r = c >= static_cast<uint64_t>(n * 8) ? 0 : ma >> c;
        break;
      }
      case O_MUL24: {
        int64_t x = s ? sext(a, 3) : static_cast<int64_t>(zext(a, 3));
        int64_t y = s ? sext(b, 3) : static_cast<int64_t>(zext(b, 3));
        int64_t p = x * y;
        r = in.hi ? static_cast<uint64_t>(p >> 16) : static_cast<uint64_t>(p);
        break;
      }
      default: throw Err("integer operation " + in.op);
    }
    return norm(r, ty);
  }

  bool compare(const Ins& in, uint64_t a, uint64_t b) {
    const TI& ty = in.t;
    int n = ty.sz;
    Cmp c = static_cast<Cmp>(in.sub);
    if (ty.fl) {
      double x = n == 4 ? f32(a) : f64(a), y = n == 4 ? f32(b) : f64(b);
      bool un = std::isnan(x) || std::isnan(y);
      switch (c) {
        case C_EQ: return !un && x == y;
        case C_NE: return !un && x != y;
        case C_LT: return !un && x < y;
        case C_LE: return !un && x <= y;
        case C_GT: return !un && x > y;
        case C_GE: return !un && x >= y;
        case C_EQU: return un || x == y;
        case C_NEU: return un || x != y;
        case C_LTU: return un || x < y;
        case C_LEU: return un || x <= y;
        case C_GTU: return un || x > y;
        case C_GEU: return un || x >= y;
        case C_NUM: return !un;
        case C_NAN: return un;
        default: throw Err("comparison " + in.op);
      }
    }
    if (ty.sg) {
      int64_t x = sext(a, n), y = sext(b, n);
      switch (c) {
        case C_EQ: return x == y;
        case C_NE: return x != y;
        case C_LT: return x < y;
        case C_LE: return x <= y;
        case C_GT: return x > y;
        case C_GE: return x >= y;
        default: break;
      }
    } else {
      uint64_t x = zext(a, n), y = zext(b, n);
      switch (c) {
        case C_EQ: return x == y;
        case C_NE: return x != y;
        case C_LT: case C_LO: return x < y;
        case C_LE: case C_LS: return x <= y;
        case C_GT: case C_HI: return x > y;
        case C_GE: case C_HS: return x >= y;
        default: break;
      }
    }
    throw Err("comparison " + in.op);
  }

  uint64_t cvt(const Ins& in, uint64_t a) {
    const TI &d = in.t, &s = in.t2;
    int dn = d.sz, sn = s.sz;
    if (s.fl) {
      double x = sn == 4 ? f32(a) : f64(a);
      if (d.fl) {
        if (in.rint) {
          RoundGuard rg(in.rint && in.rm == FE_TONEAREST ? FE_TONEAREST : in.rm);
          x = std::nearbyint(x);
        }
        if (in.sat) x = std::isnan(x) ? 0 : (x < 0 ? 0 : (x > 1 ? 1 : x));
        if (dn == 4) {
          RoundGuard rg(in.rint ? FE_TONEAREST : in.rm);
          volatile float f = static_cast<float>(x);
          return bf32(f);
        }
        return bf64(x);
      }
      double r;
      {
        RoundGuard rg(in.rint ? in.rm : FE_TOWARDZERO);
        r = std::nearbyint(x);
      }
      if (std::isnan(r)) return 0;
      if (d.sg) {
        double lo = -std::ldexp(1.0, dn * 8 - 1), hi = std::ldexp(1.0, dn * 8 - 1) - 1;
        r = r < lo ? lo : (r > hi ? hi : r);
        return norm(static_cast<uint64_t>(static_cast<int64_t>(r)), d);
      }
      double hi = std::ldexp(1.0, dn * 8) - 1;
      r = r < 0 ? 0 : (r > hi ? hi : r);
      if (dn == 8 && r >= 9223372036854775808.0) return static_cast<uint64_t>(r - 9223372036854775808.0) | (1ULL << 63);
      return static_cast<uint64_t>(r);
    }
    if (d.fl) {
      RoundGuard rg(in.rm);
      if (s.sg) {
        int64_t v = sext(a, sn);
        if (dn == 4) {
          volatile float f = static_cast<float>(v);
          return bf32(f);
        }
        volatile double f = static_cast<double>(v);
        return bf64(f);
      }
      uint64_t v = zext(a, sn);
      if (dn == 4) {
        volatile float f = static_cast<float>(v);
        return bf32(f);
      }
      volatile double f = static_cast<double>(v);
      return bf64(f);
    }
    uint64_t v = s.sg ? static_cast<uint64_t>(sext(a, sn)) : zext(a, sn);
    if (in.sat) {
      if (s.sg) {
        int64_t x = static_cast<int64_t>(v);
        if (d.sg) {
          int64_t lo = -(1LL << (dn * 8 - 1)), hi = (dn == 8 ? INT64_MAX : (1LL << (dn * 8 - 1)) - 1);
          x = x < lo ? lo : (x > hi ? hi : x);
        } else {
          uint64_t hi = dn == 8 ? ~0ULL : (1ULL << (dn * 8)) - 1;
          x = x < 0 ? 0 : (static_cast<uint64_t>(x) > hi ? static_cast<int64_t>(hi) : x);
        }
        v = static_cast<uint64_t>(x);
      } else {
        uint64_t hi = d.sg ? (dn == 8 ? static_cast<uint64_t>(INT64_MAX) : (1ULL << (dn * 8 - 1)) - 1)
                           : (dn == 8 ? ~0ULL : (1ULL << (dn * 8)) - 1);
        v = v > hi ? hi : v;
      }
    }
    return norm(v, d);
  }

  uint64_t atomic_value(const Ins& in, uint64_t old, uint64_t b, uint64_t c) {
    int n = in.t.sz;
    switch (in.sub) {
      case A_CAS: return zext(old, n) == zext(b, n) ? c : old;
      case A_EXCH: return b;
      case A_INC: {
        uint32_t o = static_cast<uint32_t>(old), lim = static_cast<uint32_t>(b);
        return o >= lim ? 0 : o + 1;
      }
      case A_DEC: {
        uint32_t o = static_cast<uint32_t>(old), lim = static_cast<uint32_t>(b);
        return (o == 0 || o > lim) ? lim : o - 1;
      }
      case A_ADD: return arith(in, O_ADD, old, b);
      case A_MIN: return arith(in, O_MIN, old, b);
      case A_MAX: return arith(in, O_MAX, old, b);
      case A_AND: return arith(in, O_AND, old, b);
      case A_OR: return arith(in, O_OR, old, b);
      case A_XOR: return arith(in, O_XOR, old, b);
    }
    return old;
  }

  uint64_t load(uint8_t* p, const TI& t) {
    uint64_t v = 0;
    memcpy(&v, p, t.sz);
    return norm(v, t);
  }

  // Executes one instruction of thread t (or marks it as waiting at a barrier/collective).
  void exec(Thread& t) {
    const Ins& in = code[t.pc];
    if (++t.icount > kMaxInstructionsPerThread) throw Err("kernel did not finish (endless loop?)");
    if (in.pred >= 0 && ((t.r[in.pred] != 0) == in.predneg)) {
      t.pc++;
      return;
    }
    const TI& ty = in.t;
    switch (in.code) {
      case O_SHFL:
      case O_VOTE:
      case O_MATCH:
      case O_ACTIVEMASK:
      case O_BARWARP: t.st = WARP; return;
      case O_BAR: t.st = BAR; return;
      case O_RET: t.st = DONE; return;
      case O_TRAP: throw Err("trap (assertion or __trap)");
      case O_BRA: t.pc = in.target; return;
      default: break;
    }
    t.pc++;
    switch (in.code) {
      case O_LD: {
        uint8_t* p = addr(t, in.o[1], in.sp, static_cast<size_t>(ty.sz) * (in.o[0].k == K_VEC ? in.o[0].nv : 1));
        if (in.o[0].k == K_VEC) {
          for (int i = 0; i < in.o[0].nv; ++i)
            if (in.o[0].vr[i] >= 0) t.r[in.o[0].vr[i]] = load(p + i * ty.sz, ty);
        } else {
          set(t, in.o[0], load(p, ty));
        }
        return;
      }
      case O_ST: {
        int nv = in.o[1].k == K_VEC ? in.o[1].nv : 1;
        uint8_t* p = addr(t, in.o[0], in.sp, static_cast<size_t>(ty.sz) * nv);
        if (in.o[1].k == K_VEC) {
          for (int i = 0; i < nv; ++i) memcpy(p + i * ty.sz, &t.r[in.o[1].vr[i]], ty.sz);
        } else {
          uint64_t v = val(t, in.o[1]);
          memcpy(p, &v, ty.sz);
        }
        return;
      }
      case O_MOV: {
        int half = ty.sz * 4;   // bits per element of {a, b}
        if (in.o[0].k == K_VEC) {   // mov.b64 {lo, hi}, x
          uint64_t v = val(t, in.o[1]);
          int bits = ty.sz * 8 / in.o[0].nv;
          for (int i = 0; i < in.o[0].nv; ++i)
            if (in.o[0].vr[i] >= 0) t.r[in.o[0].vr[i]] = (v >> (i * bits)) & ((bits == 64 ? 0 : (1ULL << bits)) - 1);
          (void)half;
          return;
        }
        if (in.o[1].k == K_VEC) {   // mov.b64 x, {lo, hi}
          int bits = ty.sz * 8 / in.o[1].nv;
          uint64_t v = 0;
          for (int i = 0; i < in.o[1].nv; ++i) v |= zext(t.r[in.o[1].vr[i]], bits / 8) << (i * bits);
          set(t, in.o[0], v);
          return;
        }
        uint64_t v = val(t, in.o[1]);
        set(t, in.o[0], ty.pr ? (v != 0) : norm(v, ty));
        return;
      }
      case O_CVT: set(t, in.o[0], cvt(in, val(t, in.o[1]))); return;
      case O_CVTA: {
        uint64_t a = val(t, in.o[1]);
        uint64_t base = 0;
        if (in.sp == S_SHARED) base = reinterpret_cast<uint64_t>(L.shared.data());
        else if (in.sp == S_LOCAL) base = reinterpret_cast<uint64_t>(t.local.data());
        uint64_t r = in.to ? a - base : a + base;
        set(t, in.o[0], ty.sz == 4 ? zext(r, 4) : r);
        return;
      }
      case O_ADD: case O_SUB: case O_MUL: case O_DIV: case O_REM: case O_MIN: case O_MAX: case O_AND:
      case O_OR: case O_XOR: case O_SHL: case O_SHR: case O_MUL24: {
        uint64_t a = val(t, in.o[1]), b = val(t, in.o[2]);
        if (ty.pr) {
          bool x = a != 0, y = b != 0;
          set(t, in.o[0], in.code == O_AND ? (x && y) : (in.code == O_OR ? (x || y) : (x != y)));
          return;
        }
        set(t, in.o[0], arith(in, in.code, a, b));
        return;
      }
      case O_MAD:
      case O_MAD24: {
        uint64_t a = val(t, in.o[1]), b = val(t, in.o[2]), c = val(t, in.o[3]);
        if (ty.fl) {   // mad.f32/f64 is a fused multiply-add (PTX ISA)
          RoundGuard rg(in.rm);
          if (ty.sz == 4) {
            volatile float r = std::fmaf(f32(a), f32(b), f32(c));
            set(t, in.o[0], bf32(r));
          } else {
            volatile double r = std::fma(f64(a), f64(b), f64(c));
            set(t, in.o[0], bf64(r));
          }
          return;
        }
        uint64_t p = arith(in, in.code == O_MAD ? O_MUL : O_MUL24, a, b);
        if (in.wide) {
          set(t, in.o[0], p + c);
          return;
        }
        Ins add = in;
        add.hi = add.wide = false;
        set(t, in.o[0], arith(add, O_ADD, p, c));
        return;
      }
      case O_FMA: {
        RoundGuard rg(in.rm);
        if (ty.sz == 4) {
          volatile float r = std::fmaf(f32(val(t, in.o[1])), f32(val(t, in.o[2])), f32(val(t, in.o[3])));
          float rv = r;
          if (in.sat) rv = std::isnan(rv) ? 0.f : (rv < 0.f ? 0.f : (rv > 1.f ? 1.f : rv));
          set(t, in.o[0], bf32(rv));
        } else {
          volatile double r = std::fma(f64(val(t, in.o[1])), f64(val(t, in.o[2])), f64(val(t, in.o[3])));
          set(t, in.o[0], bf64(r));
        }
        return;
      }
      case O_NEG: case O_ABS: case O_NOT: case O_CNOT: case O_SQRT: case O_RSQRT: case O_RCP: case O_EX2:
      case O_LG2: case O_SIN: case O_COS: case O_TANH: case O_POPC: case O_CLZ: case O_BREV: case O_BFIND: {
        uint64_t a = val(t, in.o[1]);
        int n = ty.sz;
        if (ty.pr) {
          set(t, in.o[0], a == 0);
          return;
        }
        if (ty.fl && in.code != O_POPC && in.code != O_CLZ) {
          RoundGuard rg(in.rm);
          if (n == 4) {
            float x = f32(a);
            volatile float y = 0;
            switch (in.code) {
              case O_NEG: y = -x; break;
              case O_ABS: y = std::fabs(x); break;
              case O_SQRT: y = std::sqrt(x); break;
              case O_RSQRT: y = static_cast<float>(1.0 / std::sqrt(static_cast<double>(x))); break;
              case O_RCP: y = 1.0f / x; break;
              case O_EX2: y = static_cast<float>(std::exp2(static_cast<double>(x))); break;
              case O_LG2: y = static_cast<float>(std::log2(static_cast<double>(x))); break;
              case O_SIN: y = static_cast<float>(std::sin(static_cast<double>(x))); break;
              case O_COS: y = static_cast<float>(std::cos(static_cast<double>(x))); break;
              case O_TANH: y = static_cast<float>(std::tanh(static_cast<double>(x))); break;
              default: throw Err("float operation " + in.op);
            }
            set(t, in.o[0], bf32(y));
          } else {
            double x = f64(a);
            volatile double y = 0;
            switch (in.code) {
              case O_NEG: y = -x; break;
              case O_ABS: y = std::fabs(x); break;
              case O_SQRT: y = std::sqrt(x); break;
              case O_RSQRT: y = 1.0 / std::sqrt(x); break;
              case O_RCP: y = 1.0 / x; break;
              case O_EX2: y = std::exp2(x); break;
              case O_LG2: y = std::log2(x); break;
              case O_SIN: y = std::sin(x); break;
              case O_COS: y = std::cos(x); break;
              case O_TANH: y = std::tanh(x); break;
              default: throw Err("double operation " + in.op);
            }
            set(t, in.o[0], bf64(y));
          }
          return;
        }
        uint64_t v = zext(a, n), r = 0;
        switch (in.code) {
          case O_NEG: r = zext(static_cast<uint64_t>(-static_cast<int64_t>(v)), n); break;
          case O_ABS: {
            int64_t sv = sext(a, n);
            r = zext(static_cast<uint64_t>(sv < 0 ? -sv : sv), n);
            break;
          }
          case O_NOT: r = zext(~v, n); break;
          case O_CNOT: r = v == 0 ? 1 : 0; break;
          case O_POPC: set(t, in.o[0], static_cast<uint64_t>(__builtin_popcountll(v))); return;
          case O_CLZ:
            set(t, in.o[0], v == 0 ? static_cast<uint64_t>(n * 8)
                                   : static_cast<uint64_t>(__builtin_clzll(v) - (64 - n * 8)));
            return;
          case O_BREV:
            for (int i = 0; i < n * 8; ++i)
              if (v >> i & 1) r |= 1ULL << (n * 8 - 1 - i);
            break;
          case O_BFIND: {   // position of the most significant non-sign bit
            uint64_t x = v;
            if (ty.sg && sext(a, n) < 0) x = zext(~v, n);
            int pos = x == 0 ? -1 : 63 - __builtin_clzll(x);
            uint64_t res = pos < 0 ? 0xffffffffu : static_cast<uint64_t>(in.shiftamt ? (n * 8 - 1 - pos) : pos);
            set(t, in.o[0], res);
            return;
          }
          default: throw Err("integer operation " + in.op);
        }
        set(t, in.o[0], norm(r, ty));
        return;
      }
      case O_BFE: {
        uint64_t a = val(t, in.o[1]);
        unsigned pos = static_cast<unsigned>(val(t, in.o[2]) & 0xff), len = static_cast<unsigned>(val(t, in.o[3]) & 0xff);
        int nb = ty.sz * 8;
        uint64_t x = zext(a, ty.sz), r = 0;
        bool sbit = false;
        if (ty.sg) {
          unsigned msb = len == 0 ? 0 : std::min<unsigned>(pos + len - 1, nb - 1);
          sbit = len != 0 && ((x >> msb) & 1);
        }
        for (int i = 0; i < nb; ++i) {
          bool bit = (static_cast<unsigned>(i) < len && pos + i < static_cast<unsigned>(nb)) ? ((x >> (pos + i)) & 1) : sbit;
          if (bit) r |= 1ULL << i;
        }
        set(t, in.o[0], norm(r, ty));
        return;
      }
      case O_BFI: {
        uint64_t a = val(t, in.o[1]), b = val(t, in.o[2]);
        unsigned pos = static_cast<unsigned>(val(t, in.o[3]) & 0xff), len = static_cast<unsigned>(val(t, in.o[4]) & 0xff);
        int nb = ty.sz * 8;
        uint64_t r = zext(b, ty.sz);
        for (unsigned i = 0; i < len && pos + i < static_cast<unsigned>(nb); ++i) {
          uint64_t bit = (a >> i) & 1;
          r = (r & ~(1ULL << (pos + i))) | (bit << (pos + i));
        }
        set(t, in.o[0], r);
        return;
      }
      case O_SETP: {
        bool r = compare(in, val(t, in.o[1]), val(t, in.o[2]));
        set(t, in.o[0], r);
        if (in.o[0].reg2 >= 0) t.r[in.o[0].reg2] = !r;
        return;
      }
      case O_SELP: set(t, in.o[0], val(t, in.o[3]) != 0 ? val(t, in.o[1]) : val(t, in.o[2])); return;
      case O_TESTP: {
        double x = ty.sz == 4 ? f32(val(t, in.o[1])) : f64(val(t, in.o[1]));
        bool r = false;
        switch (in.sub) {
          case M_NAN: r = std::isnan(x); break;
          case M_NUM: r = !std::isnan(x); break;
          case M_FINITE: r = std::isfinite(x); break;
          case M_INF: r = std::isinf(x); break;
          case M_NORMAL: r = ty.sz == 4 ? std::isnormal(static_cast<float>(x)) : std::isnormal(x); break;
          case M_SUBNORMAL:
            r = ty.sz == 4 ? (std::fpclassify(static_cast<float>(x)) == FP_SUBNORMAL) : (std::fpclassify(x) == FP_SUBNORMAL);
            break;
        }
        set(t, in.o[0], r);
        return;
      }
      case O_COPYSIGN: {   // copysign d, a, b: magnitude of b, sign of a
        if (ty.sz == 4) set(t, in.o[0], bf32(std::copysign(f32(val(t, in.o[2])), f32(val(t, in.o[1])))));
        else set(t, in.o[0], bf64(std::copysign(f64(val(t, in.o[2])), f64(val(t, in.o[1])))));
        return;
      }
      case O_SAD: {
        uint64_t a = val(t, in.o[1]), b = val(t, in.o[2]), c = val(t, in.o[3]);
        uint64_t d = ty.sg ? static_cast<uint64_t>(std::llabs(sext(a, ty.sz) - sext(b, ty.sz)))
                           : (zext(a, ty.sz) > zext(b, ty.sz) ? zext(a, ty.sz) - zext(b, ty.sz) : zext(b, ty.sz) - zext(a, ty.sz));
        set(t, in.o[0], norm(d + c, ty));
        return;
      }
      case O_PRMT: {
        uint64_t a = val(t, in.o[1]) & 0xffffffffu, b = val(t, in.o[2]) & 0xffffffffu, s = val(t, in.o[3]);
        uint64_t bytes = a | (b << 32), r = 0;
        for (int i = 0; i < 4; ++i) {
          unsigned sel = (s >> (4 * i)) & 0xf;
          uint64_t byte = (bytes >> ((sel & 7) * 8)) & 0xff;
          if (sel & 8) byte = (byte & 0x80) ? 0xff : 0;
          r |= byte << (8 * i);
        }
        set(t, in.o[0], r);
        return;
      }
      case O_SHF: {
        bool left = in.mods[1] == "l";
        bool clamp = in.mods[2] == "clamp";
        uint64_t lo = val(t, in.o[1]) & 0xffffffffu, hi = val(t, in.o[2]) & 0xffffffffu;
        unsigned n = static_cast<unsigned>(val(t, in.o[3]) & 0xffffffffu);
        n = clamp ? (n > 32 ? 32 : n) : (n & 31);
        uint64_t v = lo | (hi << 32);
        set(t, in.o[0], left ? ((v << n) >> 32) & 0xffffffffu : (v >> n) & 0xffffffffu);
        return;
      }
      case O_FNS: {
        uint32_t mask = static_cast<uint32_t>(val(t, in.o[1]));
        int base = static_cast<int>(val(t, in.o[2]) & 31);
        int offset = static_cast<int>(static_cast<int32_t>(val(t, in.o[3])));
        int pos = -1;
        if (offset == 0) {
          pos = (mask >> base & 1) ? base : -1;
        } else if (offset > 0) {
          int cnt = 0;
          for (int i = base; i < 32; ++i)
            if ((mask >> i & 1) && ++cnt == offset) {
              pos = i;
              break;
            }
        } else {
          int cnt = 0;
          for (int i = base; i >= 0; --i)
            if ((mask >> i & 1) && --cnt == offset) {
              pos = i;
              break;
            }
        }
        set(t, in.o[0], pos < 0 ? 0xffffffffu : static_cast<uint64_t>(pos));
        return;
      }
      case O_ATOM:
      case O_RED: {
        int di = in.code == O_ATOM ? 1 : 0;
        int n = ty.sz;
        uint8_t* p = addr(t, in.o[di], in.sp, static_cast<size_t>(n));
        uint64_t b = val(t, in.o[di + 1]);
        uint64_t c = in.sub == A_CAS ? val(t, in.o[di + 2]) : 0;
        uint64_t old = 0;
        if (n == 4) {
          uint32_t o = __atomic_load_n(reinterpret_cast<uint32_t*>(p), __ATOMIC_RELAXED);
          uint32_t want;
          do {
            want = static_cast<uint32_t>(atomic_value(in, o, b, c));
          } while (!__atomic_compare_exchange_n(reinterpret_cast<uint32_t*>(p), &o, want, true, __ATOMIC_SEQ_CST,
                                                __ATOMIC_RELAXED));
          old = o;
        } else if (n == 8) {
          uint64_t o = __atomic_load_n(reinterpret_cast<uint64_t*>(p), __ATOMIC_RELAXED);
          uint64_t want;
          do {
            want = atomic_value(in, o, b, c);
          } while (!__atomic_compare_exchange_n(reinterpret_cast<uint64_t*>(p), &o, want, true, __ATOMIC_SEQ_CST,
                                                __ATOMIC_RELAXED));
          old = o;
        } else {
          std::lock_guard<std::mutex> lk(g_atomic_mu);
          memcpy(&old, p, n);
          uint64_t nv = atomic_value(in, old, b, c);
          memcpy(p, &nv, n);
        }
        if (di) set(t, in.o[0], norm(old, ty));
        return;
      }
      case O_NOP: return;
      case O_ISSPACEP: {
        uint64_t a = val(t, in.o[1]);
        uint64_t sb = reinterpret_cast<uint64_t>(L.shared.data()), lb = reinterpret_cast<uint64_t>(t.local.data());
        bool sh = a >= sb && a < sb + L.shared.size(), lo = a >= lb && a < lb + t.local.size();
        bool r = in.sp == S_SHARED ? sh : (in.sp == S_LOCAL ? lo : (in.sp == S_GLOBAL ? (!sh && !lo) : false));
        set(t, in.o[0], r);
        return;
      }
      case O_CALL: call(t, in); return;
      default: throw Err("unsupported instruction " + in.op);
    }
  }

  void call(Thread& t, const Ins& in) {
    auto arg = [&](size_t i) {
      if (i >= in.callargs.size()) throw Err("missing argument in call to " + in.callee);
      uint64_t v = 0;
      memcpy(&v, t.cparam.data() + L.m->syms[in.callargs[i]].addr, 8);
      return v;
    };
    uint64_t ret = 0;
    if (in.callee == "vprintf") {
      ret = static_cast<uint64_t>(vprintf_impl(reinterpret_cast<const char*>(arg(0)),
                                               reinterpret_cast<const uint8_t*>(arg(1))));
    } else if (in.callee == "malloc") {
      ret = reinterpret_cast<uint64_t>(::malloc(static_cast<size_t>(arg(0))));
    } else if (in.callee == "free") {
      ::free(reinterpret_cast<void*>(arg(0)));
    }
    if (in.callret >= 0) memcpy(t.cparam.data() + L.m->syms[in.callret].addr, &ret, 8);
  }

  static int vprintf_impl(const char* fmt, const uint8_t* args) {
    std::string out;
    size_t off = 0;
    for (const char* p = fmt; *p; ++p) {
      if (*p != '%') {
        out += *p;
        continue;
      }
      if (p[1] == '%') {
        out += '%';
        ++p;
        continue;
      }
      std::string spec = "%";
      const char* q = p + 1;
      while (*q && strchr("-+ #0123456789.", *q)) spec += *q++;
      int longs = 0;
      while (*q && strchr("hlLzjt", *q)) {
        if (*q != 'h') longs++;
        spec += *q++;
      }
      char conv = *q;
      if (!conv) break;
      spec += conv;
      p = q;
      char buf[512];
      bool isf = strchr("fFeEgGaA", conv) != nullptr;
      bool eight = longs > 0 || isf || conv == 's' || conv == 'p';
      size_t sz = eight ? 8 : 4;
      off = (off + sz - 1) / sz * sz;
      uint64_t v = 0;
      if (args) memcpy(&v, args + off, sz);
      off += sz;
      if (isf) snprintf(buf, sizeof buf, spec.c_str(), f64(v));
      else if (conv == 's') snprintf(buf, sizeof buf, spec.c_str(), v ? reinterpret_cast<const char*>(v) : "(null)");
      else if (conv == 'p') snprintf(buf, sizeof buf, spec.c_str(), reinterpret_cast<void*>(v));
      else if (eight) snprintf(buf, sizeof buf, spec.c_str(), static_cast<long long>(v));
      else snprintf(buf, sizeof buf, spec.c_str(), static_cast<int>(v));
      out += buf;
    }
    std::lock_guard<std::mutex> lk(g_print_mu);
    fputs(out.c_str(), stdout);
    fflush(stdout);
    return static_cast<int>(out.size());
  }

  // Executes one warp collective whose members have all arrived. Returns true if one ran.
  bool warp_collective(int first, int last) {
    uint32_t live = 0;
    for (int i = first; i < last; ++i)
      if (T[i].st != DONE) live |= 1u << (i - first);
    std::map<int, uint32_t> bypc;
    for (int i = first; i < last; ++i)
      if (T[i].st == WARP) bypc[T[i].pc] |= 1u << (i - first);
    for (auto& kv : bypc) {
      const Ins& in = code[kv.first];
      uint32_t here = kv.second;
      if (in.code == O_ACTIVEMASK) {   // every other lane is blocked elsewhere or done
        for (int i = first; i < last; ++i)
          if (here >> (i - first) & 1) {
            T[i].r[in.o[0].reg] = here;
            T[i].pc++;
            T[i].st = RUN;
          }
        return true;
      }
      // lanes may name different member masks at the same instruction (e.g. two 16-lane tiles)
      uint32_t pending = here;
      while (pending) {
        int lead = first + __builtin_ctz(pending);
        uint32_t mask = 0xffffffffu;
        int mi = in.code == O_BARWARP ? 0 : (in.code == O_SHFL ? 4 : 2);
        bool has_mask = in.code == O_BARWARP || in.code == O_MATCH || in.sync;
        if (has_mask && mi < in.n) mask = static_cast<uint32_t>(val(T[lead], in.o[mi]));
        if (!has_mask) mask = here;   // legacy non-sync form: the lanes that arrived together
        uint32_t need = mask & live;
        uint32_t members = here & mask;
        if (!(mask >> (lead - first) & 1)) throw Err(in.op + ": the calling lane is not in its member mask");
        if ((here & need) == need) {
          execute_collective(in, first, last, members);
          return true;
        }
        pending &= ~mask;
      }
    }
    return false;
  }

  void execute_collective(const Ins& in, int first, int last, uint32_t members) {
    switch (in.code) {
      case O_BARWARP: break;
      case O_VOTE: {
        uint32_t ballot = 0;
        for (int i = first; i < last; ++i)
          if ((members >> (i - first) & 1) && val(T[i], in.o[1]) != 0) ballot |= 1u << (i - first);
        bool all = (ballot & members) == members, any = ballot != 0, uni = all || ballot == 0;
        for (int i = first; i < last; ++i) {
          if (!(members >> (i - first) & 1)) continue;
          uint64_t r = in.sub == M_BALLOT ? ballot : (in.sub == M_ALL ? all : (in.sub == M_ANY ? any : uni));
          T[i].r[in.o[0].reg] = r;
        }
        break;
      }
      case O_MATCH: {
        uint64_t v[32] = {0};
        for (int i = first; i < last; ++i)
          if (members >> (i - first) & 1) v[i - first] = zext(val(T[i], in.o[1]), in.t.sz);
        for (int i = first; i < last; ++i) {
          int li = i - first;
          if (!(members >> li & 1)) continue;
          uint32_t same = 0;
          for (int j = 0; j < 32; ++j)
            if ((members >> j & 1) && v[j] == v[li]) same |= 1u << j;
          if (in.sub == M_ALL) {
            bool eq = same == members;
            T[i].r[in.o[0].reg] = eq ? members : 0;
            if (in.o[0].reg2 >= 0) T[i].r[in.o[0].reg2] = eq;
          } else {
            T[i].r[in.o[0].reg] = same;
          }
        }
        break;
      }
      case O_SHFL: {
        uint64_t src[32] = {0};
        for (int i = first; i < last; ++i) src[i - first] = val(T[i], in.o[1]) & 0xffffffffu;
        for (int i = first; i < last; ++i) {
          int lane = i - first;
          if (!(members >> lane & 1)) continue;
          Thread& t = T[i];
          int b = static_cast<int>(val(t, in.o[2]) & 31);
          unsigned c = static_cast<unsigned>(val(t, in.o[3]));
          int cval = static_cast<int>(c & 31), seg = static_cast<int>((c >> 8) & 31);
          int maxLane = (lane & seg) | (cval & ~seg);
          int minLane = lane & seg;
          int j;
          bool pval;
          switch (in.sub) {
            case M_UP: j = lane - b; pval = j >= maxLane; break;
            case M_DOWN: j = lane + b; pval = j <= maxLane; break;
            case M_BFLY: j = lane ^ b; pval = j <= maxLane; break;
            default: j = minLane | (b & ~seg); pval = j <= maxLane; break;
          }
          if (!pval || j < 0 || j > 31) j = lane;
          t.r[in.o[0].reg] = src[j];
          if (in.o[0].reg2 >= 0) t.r[in.o[0].reg2] = pval;
        }
        break;
      }
      default: throw Err("not a warp collective: " + in.op);
    }
    for (int i = first; i < last; ++i)
      if (members >> (i - first) & 1) {
        T[i].pc++;
        T[i].st = RUN;
      }
  }

  bool barrier(int nthreads) {
    bool any = false;
    for (int i = 0; i < nthreads; ++i) {
      if (T[i].st == DONE) continue;
      if (T[i].st != BAR) return false;
      any = true;
    }
    if (!any) return false;
    // bar.red: combine the predicates of all waiting threads
    uint64_t cnt = 0;
    bool all = true, anyp = false, red = false;
    for (int i = 0; i < nthreads; ++i) {
      if (T[i].st != BAR) continue;
      const Ins& in = code[T[i].pc];
      if (in.sub < 0) continue;
      red = true;
      bool p = val(T[i], in.o[in.n - 1]) != 0;
      cnt += p;
      all = all && p;
      anyp = anyp || p;
    }
    for (int i = 0; i < nthreads; ++i) {
      if (T[i].st != BAR) continue;
      const Ins& in = code[T[i].pc];
      if (red && in.sub >= 0) T[i].r[in.o[0].reg] = in.sub == M_POPC ? cnt : (in.sub == M_AND ? all : anyp);
      T[i].pc++;
      T[i].st = RUN;
    }
    return true;
  }

  void run_block() {
    int n = static_cast<int>(T.size());
    int nwarps = (n + 31) / 32;
    for (;;) {
      bool progress = false;
      for (int w = 0; w < nwarps; ++w) {
        int first = w * 32, last = std::min(n, first + 32);
        for (;;) {   // run the warp in lock step until every lane waits or has finished
          bool any = false;
          for (int i = first; i < last; ++i)
            if (T[i].st == RUN) {
              exec(T[i]);
              any = true;
            }
          if (any) {
            progress = true;
            continue;
          }
          if (warp_collective(first, last)) {
            progress = true;
            continue;
          }
          break;
        }
      }
      bool done = true;
      for (int i = 0; i < n; ++i)
        if (T[i].st != DONE) done = false;
      if (done) return;
      if (barrier(n)) continue;
      if (!progress) {
        std::string s = "deadlock: threads wait at barriers or warp operations that can never complete (";
        int shown = 0;
        for (int i = 0; i < n && shown < 8; ++i)
          if (T[i].st != DONE) {
            s += (shown++ ? ", " : "") + std::string("thread ") + std::to_string(i) + " at '" + code[T[i].pc].op + "'";
          }
        throw Err(s + ")");
      }
    }
  }
};

// ---------------------------------------------------------------------------------------
// Driver API
// ---------------------------------------------------------------------------------------
std::mutex g_mu;
CUresult g_pending = 0;
std::vector<std::unique_ptr<Module>> g_modules;

struct Func {
  Module* m;
  Entry* e;
};
std::vector<std::unique_ptr<Func>> g_funcs;

CUresult launch(Func* f, const unsigned grid[3], const unsigned block[3], unsigned shmem, void** params) {
  unsigned long long nthreads = 1ULL * block[0] * block[1] * block[2];
  if (nthreads == 0 || nthreads > 1024 || block[0] > 1024 || block[1] > 1024 || block[2] > 64) return 1;
  if (grid[0] == 0 || grid[1] == 0 || grid[2] == 0 || grid[1] > 65535 || grid[2] > 65535) return 1;
  if (shmem + f->e->shared_static > 49152) return 1;
  Launch base;
  base.m = f->m;
  base.e = f->e;
  base.shmem = shmem;
  for (int i = 0; i < 3; ++i) {
    base.grid[i] = grid[i];
    base.block[i] = block[i];
  }
  base.params.assign(f->e->param_bytes + 16, 0);
  for (size_t i = 0; i < f->e->params.size(); ++i) {
    const Sym& s = f->m->syms[f->e->params[i]];
    if (!params || !params[i]) return 1;
    memcpy(base.params.data() + s.addr, params[i], s.size);
  }
  base.dyn_off = (f->e->shared_static + 15) / 16 * 16;
  static const bool trace = getenv("PTXSIM_TRACE") != nullptr;
  // registers and local memory start with garbage, as on a GPU (PTXSIM_POISON=0 starts them at zero)
  static const bool poison = !(getenv("PTXSIM_POISON") && getenv("PTXSIM_POISON")[0] == '0');
  const unsigned long long nblocks = 1ULL * grid[0] * grid[1] * grid[2];
  unsigned nworkers = std::thread::hardware_concurrency();
  if (const char* e = getenv("PTXSIM_THREADS")) nworkers = static_cast<unsigned>(atoi(e));
  if (nworkers < 1) nworkers = 1;
  if (nworkers > nblocks) nworkers = static_cast<unsigned>(nblocks);
  std::atomic<unsigned long long> next{0};
  std::atomic<bool> failed{false};
  std::atomic<uint64_t> executed{0};
  std::string failure;
  std::mutex fail_mu;
  auto worker = [&]() {
    Launch L = base;
    L.shared.assign(L.dyn_off + shmem + 16, 0);
    std::vector<Thread> T(nthreads);
    uint64_t count = 0;
    unsigned long long bi = 0;
    try {
      while (!failed.load(std::memory_order_relaxed) && (bi = next.fetch_add(1)) < nblocks) {
        L.ctaid[0] = static_cast<unsigned>(bi % grid[0]);
        L.ctaid[1] = static_cast<unsigned>((bi / grid[0]) % grid[1]);
        L.ctaid[2] = static_cast<unsigned>(bi / (1ULL * grid[0] * grid[1]));
        std::fill(L.shared.begin(), L.shared.end(), 0xcd);
        unsigned idx = 0;
        for (unsigned z = 0; z < block[2]; ++z)
          for (unsigned y = 0; y < block[1]; ++y)
            for (unsigned x = 0; x < block[0]; ++x, ++idx) {
              Thread& t = T[idx];
              t.r.assign(static_cast<size_t>(f->e->nregs) + 1, poison ? 0xdeadbeefcafef00dULL : 0);
              t.pc = 0;
              t.st = RUN;
              t.icount = 0;
              t.tid[0] = x;
              t.tid[1] = y;
              t.tid[2] = z;
              t.lane = static_cast<int>(idx % 32);
              t.warp = static_cast<int>(idx / 32);
              t.local.assign(f->e->local_bytes + 16, poison ? 0xcd : 0);
              t.cparam.assign(f->m->cparam_bytes + 16, 0);
            }
        Exec X(L, T);
        X.run_block();
        if (trace)
          for (const Thread& t : T) count += t.icount;
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lk(fail_mu);
      if (!failed.exchange(true)) {
        char where[96];
        snprintf(where, sizeof where, "block (%u,%u,%u): ", L.ctaid[0], L.ctaid[1], L.ctaid[2]);
        failure = where + std::string(e.what());
      }
    }
    executed += count;
  };
  if (nworkers == 1) {
    worker();
  } else {
    std::vector<std::thread> pool;
    for (unsigned w = 0; w < nworkers; ++w) pool.emplace_back(worker);
    for (std::thread& th : pool) th.join();
  }
  if (trace)
    fprintf(stderr, "ptxsim: %s grid (%u,%u,%u) block (%u,%u,%u) shared %zu+%u: %llu instructions\n",
            f->e->name.c_str(), grid[0], grid[1], grid[2], block[0], block[1], block[2], f->e->shared_static, shmem,
            static_cast<unsigned long long>(executed.load()));
  if (failed) {
    fprintf(stderr, "ptxsim: kernel %s, %s\n", f->e->name.c_str(), failure.c_str());
    std::lock_guard<std::mutex> lk(g_mu);
    g_pending = 719;   // reported at the next synchronisation, like a real kernel fault
  }
  return 0;
}

}  // namespace

extern "C" {

CUresult cuInit(unsigned) { return 0; }
CUresult cuDriverGetVersion(int* v) {
  *v = 12000;
  return 0;
}
CUresult cuDeviceGetCount(int* n) {
  *n = 1;
  return 0;
}
CUresult cuDeviceGet(int* d, int ordinal) {
  if (ordinal != 0) return 101;
  *d = 0;
  return 0;
}
CUresult cuDeviceGetName(char* name, int len, int) {
  snprintf(name, static_cast<size_t>(len), "PolyKernel PTX simulator");
  return 0;
}
CUresult cuDeviceGetAttribute(int* v, int attr, int) {
  static const std::map<int, int> a = {
      {1, 1024}, {2, 1024}, {3, 1024}, {4, 64}, {5, 2147483647}, {6, 65535}, {7, 65535}, {8, 49152},
      {9, 65536}, {10, 32}, {11, 2147483647}, {12, 65536}, {13, 1590000}, {14, 512}, {15, 1}, {16, 40},
      {19, 1}, {31, 1}, {36, 5001000}, {37, 256}, {38, 4194304}, {39, 1024}, {40, 3}, {41, 1}, {75, 7},
      {76, 5}, {78, 1}, {79, 1}, {80, 1}, {81, 65536}, {82, 65536}, {83, 1}, {89, 1}, {95, 1}, {97, 65536}};
  auto it = a.find(attr);
  *v = it == a.end() ? 0 : it->second;
  return 0;
}
CUresult cuDeviceTotalMem_v2(size_t* b, int) {
  *b = 16ULL << 30;
  return 0;
}
CUresult cuDevicePrimaryCtxRetain(void** ctx, int) {
  static int dummy;
  *ctx = &dummy;
  return 0;
}
CUresult cuCtxSetCurrent(void*) { return 0; }
CUresult cuCtxSynchronize() {
  std::lock_guard<std::mutex> lk(g_mu);
  CUresult r = g_pending;
  return r;   // a kernel fault is sticky, as on a real GPU
}
CUresult cuModuleLoadDataEx(void** mod, const void* image, unsigned n, int* opts, void** vals) {
  std::unique_ptr<Module> m(new Module);
  try {
    Parser p(*m);
    p.parse(static_cast<const char*>(image));
  } catch (const std::exception& e) {
    for (unsigned i = 0; i < n; ++i)
      if (opts[i] == 5 && vals[i]) snprintf(static_cast<char*>(vals[i]), 4096, "ptxsim: %s", e.what());
    return 218;   // CUDA_ERROR_INVALID_PTX
  }
  std::lock_guard<std::mutex> lk(g_mu);
  *mod = m.get();
  g_modules.push_back(std::move(m));
  return 0;
}
CUresult cuModuleGetFunction(void** f, void* mod, const char* name) {
  Module* m = static_cast<Module*>(mod);
  for (auto& e : m->entries)
    if (e->name == name) {
      std::lock_guard<std::mutex> lk(g_mu);
      g_funcs.emplace_back(new Func{m, e.get()});
      *f = g_funcs.back().get();
      return 0;
    }
  return 500;
}
CUresult cuModuleGetGlobal_v2(CUdeviceptr* d, size_t* sz, void* mod, const char* name) {
  Module* m = static_cast<Module*>(mod);
  auto it = m->symidx.find(name);
  if (it == m->symidx.end()) return 500;
  const Sym& s = m->syms[it->second];
  if (s.space != S_GLOBAL && s.space != S_CONST) return 500;
  if (d) *d = s.addr;
  if (sz) *sz = s.size;
  return 0;
}
CUresult cuMemAlloc_v2(CUdeviceptr* d, size_t n) {
  void* p = aligned_alloc(256, (n + 255) / 256 * 256);
  if (!p) return 2;
  memset(p, 0xab, n);   // device memory starts with garbage
  *d = reinterpret_cast<CUdeviceptr>(p);
  return 0;
}
CUresult cuMemAllocManaged(CUdeviceptr* d, size_t n, unsigned) { return cuMemAlloc_v2(d, n); }
CUresult cuMemFree_v2(CUdeviceptr d) {
  free(reinterpret_cast<void*>(d));
  return 0;
}
CUresult cuMemHostAlloc(void** p, size_t n, unsigned) {
  *p = aligned_alloc(256, (n + 255) / 256 * 256);
  return *p ? 0 : 2;
}
CUresult cuMemFreeHost(void* p) {
  free(p);
  return 0;
}
CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr* d, void* h, unsigned) {
  *d = reinterpret_cast<CUdeviceptr>(h);
  return 0;
}
CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t n) {
  if (g_pending) return g_pending;
  memmove(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), n);
  return 0;
}
CUresult cuMemsetD8_v2(CUdeviceptr d, unsigned char v, size_t n) {
  if (g_pending) return g_pending;
  memset(reinterpret_cast<void*>(d), v, n);
  return 0;
}
CUresult cuMemGetInfo_v2(size_t* f, size_t* t) {
  *f = 8ULL << 30;
  *t = 16ULL << 30;
  return 0;
}
CUresult cuLaunchKernel(void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                        unsigned shmem, void*, void** params, void**) {
  if (g_pending) return g_pending;
  unsigned grid[3] = {gx, gy, gz}, block[3] = {bx, by, bz};
  return launch(static_cast<Func*>(f), grid, block, shmem, params);
}
CUresult cuGetErrorString(CUresult r, const char** s) {
  static const std::map<int, const char*> m = {
      {0, "no error"}, {1, "invalid argument"}, {2, "out of memory"}, {101, "invalid device ordinal"},
      {218, "a PTX JIT compilation failed"}, {500, "named symbol not found"},
      {719, "unspecified launch failure"}};
  auto it = m.find(r);
  *s = it == m.end() ? "unknown error" : it->second;
  return 0;
}

}  // extern "C"
