// bitlearn's Tsetlin Machine kernels (bitlearn 0.2.0, bitlearn/_core.py) written in CUDA.
//
// Built with PolyKernel this one source runs on any CPU architecture and on OpenCL
// GPUs. The program loads a case exported by export_case.py and checks, bit for bit:
//   * inference: vote totals == bitlearn's decision_function (two kernels: plain and
//     with the include masks cached in dynamic __shared__ memory);
//   * training: one epoch from the same model, data order, random pool and per-class
//     seeds gives exactly the same automaton states and clause weights as bitlearn's
//     fit_epoch, for boosted and for weighted, non-boosted feedback.
//
//   pkcc -O2 -o bitlearn_tm bitlearn_tm.cu && ./bitlearn_tm tm_case
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

typedef unsigned long long u64;

// ---------------------------------------------------------------- device code
// xorshift64*: identical sequence to bitlearn's _rng
__device__ u64 tm_rng(u64* state) {
  u64 s = *state;
  s ^= s >> 12;
  s ^= s << 25;
  s ^= s >> 27;
  *state = s;
  return s * 0x2545F4914F6CDD1DULL;
}

__device__ double tm_unit(u64 r) { return (double)(r >> 11) * (1.0 / 9007199254740992.0); }

// Saturating +1 on every counter selected by carry (bit-sliced over n_bits planes)
__device__ void tm_increment(u64* tj, u64* carry, int n_bits, int n_words) {
  for (int b = 0; b < n_bits; ++b) {
    u64* row = tj + b * n_words;
    u64 left = 0;
    for (int w = 0; w < n_words; ++w) {
      u64 cw = carry[w];
      u64 t = row[w];
      row[w] = t ^ cw;
      cw = t & cw;
      carry[w] = cw;
      left |= cw;
    }
    if (left == 0) return;
  }
  for (int b = 0; b < n_bits; ++b) {
    u64* row = tj + b * n_words;
    for (int w = 0; w < n_words; ++w) row[w] |= carry[w];
  }
}

// Saturating -1 on every counter selected by carry
__device__ void tm_decrement(u64* tj, u64* carry, int n_bits, int n_words) {
  for (int b = 0; b < n_bits; ++b) {
    u64* row = tj + b * n_words;
    u64 left = 0;
    for (int w = 0; w < n_words; ++w) {
      u64 cw = carry[w];
      u64 t = row[w];
      row[w] = t ^ cw;
      cw = (~t) & cw;
      carry[w] = cw;
      left |= cw;
    }
    if (left == 0) return;
  }
  for (int b = 0; b < n_bits; ++b) {
    u64* row = tj + b * n_words;
    for (int w = 0; w < n_words; ++w) row[w] &= ~carry[w];
  }
}

// One training epoch. One thread per class: each class owns its clauses (no shared
// writes), exactly like bitlearn's prange over classes.
__global__ void tm_fit_epoch(u64* ta, int* weights, const u64* xp, const long long* y, const long long* order,
                             const u64* word_mask, const u64* pool, int pool_len, int T, int boost, int weighted,
                             int max_weight, const u64* seeds, int n_classes, int n_clauses, int n_bits,
                             int n_words, int n, unsigned char* out_scratch, u64* carry_scratch) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_classes) return;
  u64 rs = seeds[c];
  u64* tac = ta + (size_t)c * n_clauses * n_bits * n_words;
  int* wc = weights + (size_t)c * n_clauses;
  unsigned char* out = out_scratch + (size_t)c * n_clauses;
  u64* carry = carry_scratch + (size_t)c * n_words;
  u64 pool_span = (u64)(pool_len - n_words);
  double neg_p = n_classes > 2 ? 1.0 / (n_classes - 1) : 1.0;
  double inv_2t = 1.0 / (2.0 * T);
  int top = n_bits - 1;

  for (int ii = 0; ii < n; ++ii) {
    long long i = order[ii];
    bool target;
    if (y[i] == c) {
      target = true;
    } else {
      if (neg_p < 1.0) {
        u64 r = tm_rng(&rs);
        if (tm_unit(r) >= neg_p) continue;
      }
      target = false;
    }
    const u64* x = xp + i * n_words;

    // clause outputs and the class vote
    int s = 0;
    for (int j = 0; j < n_clauses; ++j) {
      const u64* inc = tac + ((size_t)j * n_bits + top) * n_words;
      int o = 1;
      for (int w = 0; w < n_words; ++w) {
        u64 iw = inc[w];
        if ((iw & x[w]) != iw) {
          o = 0;
          break;
        }
      }
      out[j] = (unsigned char)o;
      if (o) {
        if (j & 1)
          s -= wc[j];
        else
          s += wc[j];
      }
    }
    if (s > T)
      s = T;
    else if (s < -T)
      s = -T;
    double p = target ? (T - s) * inv_2t : (T + s) * inv_2t;
    if (p <= 0.0) continue;

    for (int j = 0; j < n_clauses; ++j) {
      u64 r = tm_rng(&rs);
      if (tm_unit(r) >= p) continue;
      u64* tj = tac + (size_t)j * n_bits * n_words;
      bool positive = (j & 1) == 0;
      if (positive == target) {   // Type I feedback
        r = tm_rng(&rs);
        int off = (int)(r % pool_span);
        if (out[j]) {
          if (weighted && wc[j] < max_weight) wc[j] += 1;
          if (boost) {
            for (int w = 0; w < n_words; ++w) carry[w] = x[w];
          } else {
            r = tm_rng(&rs);
            int off2 = (int)(r % pool_span);
            for (int w = 0; w < n_words; ++w) carry[w] = x[w] & ~pool[off2 + w];
          }
          tm_increment(tj, carry, n_bits, n_words);
          for (int w = 0; w < n_words; ++w) carry[w] = (~x[w]) & pool[off + w] & word_mask[w];
          tm_decrement(tj, carry, n_bits, n_words);
        } else {
          for (int w = 0; w < n_words; ++w) carry[w] = pool[off + w];
          tm_decrement(tj, carry, n_bits, n_words);
        }
      } else if (out[j]) {        // Type II feedback
        if (weighted && wc[j] > 1) wc[j] -= 1;
        const u64* inc = tj + top * n_words;
        for (int w = 0; w < n_words; ++w) carry[w] = (~x[w]) & (~inc[w]) & word_mask[w];
        tm_increment(tj, carry, n_bits, n_words);
      }
    }
  }
}

// Inference: vote total of class blockIdx.y for sample i. Empty clauses do not vote.
__global__ void tm_class_sums(const u64* ta, const int* weights, const u64* xp, int* sums, int n, int n_classes,
                              int n_clauses, int n_bits, int n_words) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y;
  if (i >= n) return;
  const u64* x = xp + (size_t)i * n_words;
  int top = n_bits - 1;
  int s = 0;
  for (int j = 0; j < n_clauses; ++j) {
    const u64* inc = ta + (((size_t)c * n_clauses + j) * n_bits + top) * n_words;
    u64 any = 0;
    int fires = 1;
    for (int w = 0; w < n_words; ++w) {
      u64 iw = inc[w];
      any |= iw;
      if ((iw & x[w]) != iw) {
        fires = 0;
        break;
      }
    }
    if (fires && any) s += (j & 1) ? -weights[c * n_clauses + j] : weights[c * n_clauses + j];
  }
  sums[(size_t)i * n_classes + c] = s;
}

// Same result, with the class's include masks and signed weights staged once per
// block in dynamic shared memory: [n_clauses * n_words] u64 masks + [n_clauses] ints.
__global__ void tm_class_sums_shared(const u64* ta, const int* weights, const u64* xp, int* sums, int n,
                                     int n_classes, int n_clauses, int n_bits, int n_words) {
  extern __shared__ u64 smem[];
  u64* masks = smem;
  int* sw = (int*)(smem + n_clauses * n_words);
  int c = blockIdx.y;
  int top = n_bits - 1;
  for (int k = threadIdx.x; k < n_clauses * n_words; k += blockDim.x) {
    int j = k / n_words, w = k % n_words;
    masks[k] = ta[(((size_t)c * n_clauses + j) * n_bits + top) * n_words + w];
  }
  for (int j = threadIdx.x; j < n_clauses; j += blockDim.x)
    sw[j] = (j & 1) ? -weights[c * n_clauses + j] : weights[c * n_clauses + j];
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const u64* x = xp + (size_t)i * n_words;
  int s = 0;
  for (int j = 0; j < n_clauses; ++j) {
    const u64* inc = masks + j * n_words;
    u64 any = 0, miss = 0;
    for (int w = 0; w < n_words; ++w) {
      any |= inc[w];
      miss |= inc[w] & ~x[w];
    }
    if (any && !miss) s += sw[j];
  }
  sums[(size_t)i * n_classes + c] = s;
}

// ------------------------------------------------------------------ host code
#define CHECK(x)                                                                         \
  do {                                                                                   \
    cudaError_t e_ = (x);                                                                \
    if (e_ != cudaSuccess) {                                                             \
      printf("FAIL: %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);          \
      exit(1);                                                                           \
    }                                                                                    \
  } while (0)

static bool big_endian() {
  const uint16_t v = 1;
  return *reinterpret_cast<const uint8_t*>(&v) == 0;
}

template <class T>
static std::vector<T> load(const std::string& dir, const std::string& name, size_t count) {
  std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
  if (!f) {
    printf("FAIL: cannot open %s/%s.bin\n", dir.c_str(), name.c_str());
    exit(1);
  }
  std::vector<T> v(count);
  f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(count * sizeof(T)));
  if ((size_t)f.gcount() != count * sizeof(T)) {
    printf("FAIL: %s.bin has the wrong size\n", name.c_str());
    exit(1);
  }
  if (big_endian()) {   // files are little-endian
    for (T& e : v) {
      unsigned char* b = reinterpret_cast<unsigned char*>(&e);
      for (size_t k = 0; k < sizeof(T) / 2; ++k) std::swap(b[k], b[sizeof(T) - 1 - k]);
    }
  }
  return v;
}

template <class T>
static T* to_device(const std::vector<T>& v) {
  T* d = nullptr;
  CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
  CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

template <class T>
static std::vector<T> to_host(const T* d, size_t n) {
  std::vector<T> v(n);
  CHECK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "tm_case";
  std::map<std::string, std::string> meta;
  {
    std::ifstream f(dir + "/meta.txt");
    std::string line;
    while (std::getline(f, line)) {
      size_t eq = line.find('=');
      if (eq != std::string::npos) meta[line.substr(0, eq)] = line.substr(eq + 1);
    }
  }
  if (meta.empty()) {
    printf("FAIL: %s/meta.txt not found (run export_case.py first)\n", dir.c_str());
    return 1;
  }
  auto I = [&](const char* k) { return atoi(meta[k].c_str()); };
  const int K = I("n_classes"), C = I("n_clauses"), B = I("n_bits"), W = I("n_words");
  const int ntr = I("n_train"), nte = I("n_test");
  const size_t ta_n = (size_t)K * C * B * W;
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("device: %s\n", prop.name);
  printf("model: %d classes x %d clauses, %d-bit automata, %d literal words; bitlearn %s (Python %s)\n", K, C, B, W,
         meta["bitlearn_version"].c_str(), meta["python"].c_str());

  auto ta0 = load<u64>(dir, "ta0", ta_n);
  auto w0 = load<int>(dir, "w0", (size_t)K * C);
  auto xp_tr = load<u64>(dir, "xp_train", (size_t)ntr * W);
  auto y_tr = load<long long>(dir, "y_train", ntr);
  auto xp_te = load<u64>(dir, "xp_test", (size_t)nte * W);
  auto y_te = load<long long>(dir, "y_test", nte);
  auto want_sums = load<int>(dir, "sums", (size_t)nte * K);
  auto mask = load<u64>(dir, "mask", W);
  bool ok = true;

  // ---- inference
  u64* d_ta = to_device(ta0);
  int* d_w = to_device(w0);
  u64* d_xte = to_device(xp_te);
  int* d_sums = nullptr;
  CHECK(cudaMalloc(&d_sums, (size_t)nte * K * sizeof(int)));
  for (int variant = 0; variant < 2; ++variant) {
    CHECK(cudaMemset(d_sums, 0, (size_t)nte * K * sizeof(int)));
    dim3 grid((nte + 127) / 128, K);
    double t0 = now_ms();
    if (variant == 0) {
      tm_class_sums<<<grid, 128>>>(d_ta, d_w, d_xte, d_sums, nte, K, C, B, W);
    } else {
      size_t smem = (size_t)C * W * sizeof(u64) + (size_t)C * sizeof(int);
      tm_class_sums_shared<<<grid, 128, smem>>>(d_ta, d_w, d_xte, d_sums, nte, K, C, B, W);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double t1 = now_ms();
    auto sums = to_host(d_sums, (size_t)nte * K);
    size_t bad = 0;
    int correct = 0;
    for (int i = 0; i < nte; ++i) {
      int best = 0;
      for (int c = 0; c < K; ++c) {
        if (sums[(size_t)i * K + c] != want_sums[(size_t)i * K + c]) ++bad;
        if (sums[(size_t)i * K + c] > sums[(size_t)i * K + best]) best = c;
      }
      correct += (best == y_te[i]);
    }
    printf("inference (%s): %zu of %zu vote totals differ from bitlearn; accuracy %.4f (bitlearn %s); %.2f ms\n",
           variant == 0 ? "plain " : "shared", bad, (size_t)nte * K, (double)correct / nte,
           meta["bitlearn_accuracy"].c_str(), t1 - t0);
    ok = ok && bad == 0;
  }

  // ---- one training epoch, both feedback modes
  u64* d_xtr = to_device(xp_tr);
  long long* d_y = to_device(y_tr);
  u64* d_mask = to_device(mask);
  unsigned char* d_out = nullptr;
  u64* d_carry = nullptr;
  CHECK(cudaMalloc(&d_out, (size_t)K * C));
  CHECK(cudaMalloc(&d_carry, (size_t)K * W * sizeof(u64)));
  for (const char* tag : {"a", "b"}) {
    std::string t(tag);
    int pool_len = I(("pool_len_" + t).c_str());
    auto order = load<long long>(dir, "order_" + t, ntr);
    auto pool = load<u64>(dir, "pool_" + t, pool_len);
    auto seeds = load<u64>(dir, "seeds_" + t, K);
    auto want_ta = load<u64>(dir, "ta1_" + t, ta_n);
    auto want_w = load<int>(dir, "w1_" + t, (size_t)K * C);
    CHECK(cudaMemcpy(d_ta, ta0.data(), ta_n * sizeof(u64), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w, w0.data(), (size_t)K * C * sizeof(int), cudaMemcpyHostToDevice));
    long long* d_order = to_device(order);
    u64* d_pool = to_device(pool);
    u64* d_seeds = to_device(seeds);
    double t0 = now_ms();
    tm_fit_epoch<<<K, 1>>>(d_ta, d_w, d_xtr, d_y, d_order, d_mask, d_pool, pool_len, I(("T_" + t).c_str()),
                           I(("boost_" + t).c_str()), I(("weighted_" + t).c_str()), I(("max_weight_" + t).c_str()),
                           d_seeds, K, C, B, W, ntr, d_out, d_carry);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double t1 = now_ms();
    auto ta1 = to_host(d_ta, ta_n);
    auto w1 = to_host(d_w, (size_t)K * C);
    size_t bad_ta = 0, bad_w = 0, changed = 0;
    for (size_t k = 0; k < ta_n; ++k) {
      bad_ta += ta1[k] != want_ta[k];
      changed += ta1[k] != ta0[k];
    }
    for (size_t k = 0; k < (size_t)K * C; ++k) bad_w += w1[k] != want_w[k];
    printf("training epoch %s (T=%s boost=%s weighted=%s): %zu state words changed, %zu differ from bitlearn, "
           "%zu weights differ; %.1f ms\n",
           tag, meta["T_" + t].c_str(), meta["boost_" + t].c_str(), meta["weighted_" + t].c_str(), changed,
           bad_ta, bad_w, t1 - t0);
    ok = ok && bad_ta == 0 && bad_w == 0 && changed > 0;
    CHECK(cudaFree(d_order));
    CHECK(cudaFree(d_pool));
    CHECK(cudaFree(d_seeds));
  }
  printf(ok ? "PASS bitlearn_tm: CUDA port matches bitlearn bit for bit\n" : "FAIL bitlearn_tm\n");
  return ok ? 0 : 1;
}
