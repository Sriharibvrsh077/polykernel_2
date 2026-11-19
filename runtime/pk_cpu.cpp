// PolyKernel CPU back end.
//
// Execution model:
//  * A launch is split into blocks; blocks are distributed over a pool of
//    worker threads (one per hardware thread, override with PK_NUM_THREADS).
//  * Inside a block, CUDA threads run either
//      - as a plain loop (kernels in translation units that never use
//        barriers or warp intrinsics), or
//      - as cooperative fibers: each CUDA thread has its own small stack and
//        __syncthreads()/warp operations switch to the next fiber, so all
//        threads of the block reach the barrier before any continues.
//  * __shared__ variables become thread_local storage of the worker thread,
//    which is exactly "one copy per block" because a block never migrates.
//
// Fiber switching uses _setjmp/_longjmp (no system calls); new fibers are
// entered once through makecontext/setcontext. On Windows native fibers are used.

#undef _FORTIFY_SOURCE   // __longjmp_chk rejects jumps between stacks
#include "pk_internal.h"

#include <setjmp.h>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#define PK_WIN_FIBERS 1
#else
#include <sys/mman.h>
#include <ucontext.h>
#include <unistd.h>
#define PK_WIN_FIBERS 0
#endif

namespace pk {
namespace cpu {

__thread Lane* tl_lane = nullptr;
__thread Block* tl_block = nullptr;

namespace {

struct Fiber {
#if PK_WIN_FIBERS
  void* handle = nullptr;
#else
  jmp_buf jb;
  ucontext_t uc;
  void* mem = nullptr;
  size_t mem_size = 0;
#endif
  bool started = false;
  bool done = true;
  Lane lane{};
};

struct WarpBar {
  unsigned key = 0;
  unsigned arrived = 0;
  unsigned gen = 0;
};
struct Rendezvous {   // lanes meeting at one coalesced_threads()/__activemask() call site
  uintptr_t site = 0;
  unsigned pending = 0;
  unsigned gen = 0;
  unsigned result = 0;
};
struct Warp {
  unsigned live = 0;
  unsigned waiting = 0;   // lanes blocked inside a barrier wait
  WarpBar bars[32];       // a lane waits on at most one barrier, so 32 entries always suffice
  Rendezvous rv[32];
  uint64_t slots[32];
};

struct Job {
  void (*thunk)(void*);
  void* ctx;
  dim3 grid, block;
  size_t shmem;
  bool fibers;
  unsigned nthreads;
  unsigned long long nblocks;
};

struct Worker {
  std::vector<Fiber*> fibers;
#if PK_WIN_FIBERS
  void* sched_fiber = nullptr;
#else
  jmp_buf sched_jb;
#endif
  Fiber* cur = nullptr;
  const Job* job = nullptr;
  Block block{};
  bool fiber_mode = false;
  unsigned long long progress = 0;   // bumped whenever a thread arrives at a barrier or finishes
  unsigned live = 0, bar_arrived = 0, bar_gen = 0;
  int bar_acc[2] = {0, 0};
  std::vector<Warp> warps;
  unsigned char* smem = nullptr;
  size_t smem_cap = 0;
  ~Worker() {
    for (Fiber* f : fibers) {
#if PK_WIN_FIBERS
      if (f->handle) DeleteFiber(f->handle);
#else
      if (f->mem) munmap(f->mem, f->mem_size);
#endif
      delete f;
    }
    ::free(smem);
  }
};

thread_local std::unique_ptr<Worker> tl_worker_owner;
__thread Worker* tl_worker = nullptr;

Worker* get_worker() {
  if (!tl_worker) {
    tl_worker_owner.reset(new Worker());
    tl_worker = tl_worker_owner.get();
  }
  return tl_worker;
}

size_t stack_size() {
  static size_t s = env_size("PK_STACK_SIZE", sizeof(void*) == 8 ? 256 * 1024 : 64 * 1024);
  return s < 16 * 1024 ? 16 * 1024 : s;
}

bool force_fibers() {
  static int f = [] {
    const char* e = getenv("PK_FORCE_FIBERS");
    return (e && *e && *e != '0') ? 1 : 0;
  }();
  return f != 0;
}

void thread_finished(Worker* w, Fiber* f);

#if PK_WIN_FIBERS
void WINAPI win_fiber_entry(void* p) {
  Fiber* f = static_cast<Fiber*>(p);
  Worker* w = tl_worker;
  for (;;) {
    w->job->thunk(w->job->ctx);
    thread_finished(w, f);
    SwitchToFiber(w->sched_fiber);
  }
}
void resume(Worker* w, Fiber* f) {
  w->cur = f;
  tl_lane = &f->lane;
  if (!w->sched_fiber) w->sched_fiber = ConvertThreadToFiber(nullptr);
  if (!f->handle) {
    f->handle = CreateFiber(stack_size(), win_fiber_entry, f);
    if (!f->handle) {
      fprintf(stderr, "polykernel: CreateFiber failed\n");
      abort();
    }
  }
  f->started = true;
  SwitchToFiber(f->handle);
}
__attribute__((noinline)) void yield_to_sched(Worker* w, Fiber*) { SwitchToFiber(w->sched_fiber); }
void ensure_fiber_storage(Fiber*) {}
#else
void fiber_entry() {
  Worker* w = tl_worker;
  Fiber* f = w->cur;
  for (;;) {
    w->job->thunk(w->job->ctx);
    thread_finished(w, f);
    if (!_setjmp(f->jb)) _longjmp(w->sched_jb, 1);
  }
}
void ensure_fiber_storage(Fiber* f) {
  if (f->mem) return;
  size_t page = (size_t)sysconf(_SC_PAGESIZE);
  size_t ss = (stack_size() + page - 1) / page * page;
  size_t total = ss + page;
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
  flags |= MAP_NORESERVE;
#endif
#ifdef MAP_STACK
  flags |= MAP_STACK;
#endif
  void* mem = mmap(nullptr, total, PROT_READ | PROT_WRITE, flags, -1, 0);
  if (mem == MAP_FAILED) {
    fprintf(stderr, "polykernel: cannot allocate fiber stack (%zu bytes)\n", total);
    abort();
  }
  mprotect(mem, page, PROT_NONE);   // guard page: stack overflow faults instead of corrupting memory
  f->mem = mem;
  f->mem_size = total;
  getcontext(&f->uc);
  f->uc.uc_stack.ss_sp = static_cast<char*>(mem) + page;
  f->uc.uc_stack.ss_size = ss;
  f->uc.uc_link = nullptr;
  makecontext(&f->uc, (void (*)())fiber_entry, 0);
}
void resume(Worker* w, Fiber* f) {
  w->cur = f;
  tl_lane = &f->lane;
  if (!_setjmp(w->sched_jb)) {
    if (!f->started) {
      f->started = true;
      setcontext(&f->uc);
    }
    _longjmp(f->jb, 1);
  }
}
__attribute__((noinline)) void yield_to_sched(Worker* w, Fiber* f) {
  if (!_setjmp(f->jb)) _longjmp(w->sched_jb, 1);
}
#endif

void release_block_barrier(Worker* w) {
  unsigned g = w->bar_gen;
  w->bar_arrived = 0;
  w->bar_acc[(g + 1) & 1] = 0;
  w->bar_gen = g + 1;
}

void check_warp_release(Warp& wp) {
  for (WarpBar& b : wp.bars) {
    if (b.arrived && (b.arrived & wp.live) == (b.key & wp.live)) {
      b.arrived = 0;
      ++b.gen;
    }
  }
}

void thread_finished(Worker* w, Fiber* f) {
  ++w->progress;
  f->done = true;
  --w->live;
  if (w->bar_arrived && w->bar_arrived >= w->live) release_block_barrier(w);
  Warp& wp = w->warps[f->lane.linear >> 5];
  wp.live &= ~(1u << (f->lane.linear & 31u));
  check_warp_release(wp);
}

Worker* need_fibers(const char* what) {
  Worker* w = tl_worker;
  if (!w || !w->fiber_mode) {
    fprintf(stderr,
            "polykernel: %s was called from a kernel that was compiled without barrier support.\n"
            "  (pkcc found no barrier or warp intrinsic in that translation unit; this happens when the\n"
            "  call sits in a different .cu file). Re-run with PK_FORCE_FIBERS=1.\n",
            what);
    abort();
  }
  return w;
}

void ensure_fibers(Worker* w, unsigned n) {
  while (w->fibers.size() < n) w->fibers.push_back(new Fiber());
  for (unsigned i = 0; i < n; ++i) ensure_fiber_storage(w->fibers[i]);
}

void run_block(Worker* w, const Job& J, unsigned long long lb) {
  Block& b = w->block;
  unsigned long long gx = J.grid.x, gy = J.grid.y;
  b.bid.x = (unsigned)(lb % gx);
  b.bid.y = (unsigned)((lb / gx) % gy);
  b.bid.z = (unsigned)(lb / (gx * gy));
  b.bdim = J.block;
  b.gdim = J.grid;
  b.dyn_smem = w->smem;
  b.nthreads = J.nthreads;
  tl_block = &b;
  const unsigned bx = J.block.x, by = J.block.y, bz = J.block.z, n = J.nthreads;

  if (!J.fibers) {
    w->fiber_mode = false;
    Lane lane;
    tl_lane = &lane;
    unsigned lin = 0;
    for (unsigned z = 0; z < bz; ++z)
      for (unsigned y = 0; y < by; ++y)
        for (unsigned x = 0; x < bx; ++x) {
          lane.tid = uint3{x, y, z};
          lane.linear = lin++;
          J.thunk(J.ctx);
        }
    tl_lane = nullptr;
    return;
  }

  w->fiber_mode = true;
  ensure_fibers(w, n);
  w->live = n;
  w->bar_arrived = 0;
  w->bar_gen = 0;
  w->bar_acc[0] = w->bar_acc[1] = 0;
  unsigned nw = (n + 31) / 32;
  if (w->warps.size() < nw) w->warps.resize(nw);
  for (unsigned i = 0; i < nw; ++i) {
    unsigned cnt = (i + 1) * 32 <= n ? 32 : n - i * 32;
    Warp& wp = w->warps[i];
    wp.live = cnt == 32 ? 0xffffffffu : ((1u << cnt) - 1u);
    wp.waiting = 0;
    for (WarpBar& wb : wp.bars) wb = WarpBar();
    for (Rendezvous& r : wp.rv) r = Rendezvous();
  }
  for (unsigned i = 0; i < n; ++i) {
    Fiber* f = w->fibers[i];
    f->done = false;
    f->lane.linear = i;
    f->lane.tid = uint3{i % bx, (i / bx) % by, i / (bx * by)};
  }
  while (w->live > 0) {
    unsigned long long before = w->progress;
    for (unsigned i = 0; i < n; ++i) {
      Fiber* f = w->fibers[i];
      if (!f->done) resume(w, f);
    }
    if (w->progress == before && w->live > 0) {
      // every remaining thread is waiting at a barrier that can never complete
      fprintf(stderr,
              "polykernel: deadlock in block (%u,%u,%u): %u threads wait at barriers that can never complete.\n"
              "  Typical causes: __syncthreads() inside a branch not taken by every thread, or warp operations\n"
              "  whose masks disagree (e.g. coalesced_threads()/__activemask() in divergent code).\n",
              b.bid.x, b.bid.y, b.bid.z, w->live);
      abort();
    }
  }
  w->fiber_mode = false;
  tl_lane = nullptr;
}

class Pool {
 public:
  static Pool& get() {
    static Pool* p = new Pool();   // intentionally leaked: workers live for the process
    return *p;
  }
  unsigned size() const { return (unsigned)threads_.size() + 1; }

  void run(const Job& J) {
    std::lock_guard<std::mutex> launch_lock(launch_m_);
    if (threads_.empty() || J.nblocks == 1) {
      process(J);
      return;
    }
    next_.store(0);
    {
      std::lock_guard<std::mutex> l(m_);
      job_ = &J;
      pending_ = (unsigned)threads_.size();
      ++seq_;
    }
    cv_start_.notify_all();
    process(J);
    std::unique_lock<std::mutex> l(m_);
    cv_done_.wait(l, [this] { return pending_ == 0; });
    job_ = nullptr;
  }

 private:
  Pool() {
    unsigned n = (unsigned)env_size("PK_NUM_THREADS", 0);
    if (n == 0) n = std::thread::hardware_concurrency();
    if (n == 0) n = 1;
    for (unsigned i = 1; i < n; ++i) threads_.emplace_back([this] { worker_main(); });
    for (auto& t : threads_) t.detach();
  }
  void process(const Job& J) {
    Worker* w = get_worker();
    if (J.shmem > w->smem_cap) {
      ::free(w->smem);
      size_t cap = (J.shmem + 63) & ~(size_t)63;
#if defined(_WIN32)
      w->smem = static_cast<unsigned char*>(::malloc(cap + 64));
#else
      void* p = nullptr;
      if (posix_memalign(&p, 64, cap) != 0) p = nullptr;
      w->smem = static_cast<unsigned char*>(p);
#endif
      w->smem_cap = w->smem ? cap : 0;
    }
    w->job = &J;
    if (threads_.empty() || J.nblocks == 1) {
      for (unsigned long long b = 0; b < J.nblocks; ++b) run_block(w, J, b);
    } else {
      for (;;) {
        unsigned long long b = next_.fetch_add(1);
        if (b >= J.nblocks) break;
        run_block(w, J, b);
      }
    }
    w->job = nullptr;
    tl_block = nullptr;
  }
  void worker_main() {
    uint64_t seen = 0;
    for (;;) {
      const Job* J;
      {
        std::unique_lock<std::mutex> l(m_);
        cv_start_.wait(l, [&] { return seq_ != seen; });
        seen = seq_;
        J = job_;
      }
      if (J) process(*J);
      std::lock_guard<std::mutex> l(m_);
      if (--pending_ == 0) cv_done_.notify_one();
    }
  }

  std::vector<std::thread> threads_;
  std::mutex launch_m_, m_;
  std::condition_variable cv_start_, cv_done_;
  const Job* job_ = nullptr;
  uint64_t seq_ = 0;
  unsigned pending_ = 0;
  std::atomic<unsigned long long> next_{0};
};

}  // namespace

int num_threads() { return (int)Pool::get().size(); }

void syncthreads() {
  Worker* w = need_fibers("__syncthreads()");
  ++w->progress;
  unsigned g = w->bar_gen;
  if (++w->bar_arrived >= w->live) {
    release_block_barrier(w);
    return;
  }
  Fiber* f = w->cur;
  Warp& wp = w->warps[tl_lane->linear >> 5];
  unsigned bit = 1u << (tl_lane->linear & 31u);
  wp.waiting |= bit;
  do yield_to_sched(w, f);
  while (w->bar_gen == g);
  wp.waiting &= ~bit;
}

int syncthreads_count(int pred) {
  Worker* w = need_fibers("__syncthreads_count/and/or()");
  unsigned g = w->bar_gen;
  w->bar_acc[g & 1] += pred ? 1 : 0;
  syncthreads();
  return w->bar_acc[g & 1];
}

void warp_sync(unsigned mask) {
  Worker* w = need_fibers("a warp intrinsic (__syncwarp/__shfl*/__ballot*)");
  Lane* l = tl_lane;
  Warp& wp = w->warps[l->linear >> 5];
  unsigned bit = 1u << (l->linear & 31u);
  unsigned key = mask | bit;
  if (((key & wp.live) & ~bit) == 0) return;   // no other live participant
  WarpBar* b = nullptr;
  for (WarpBar& c : wp.bars)
    if (c.arrived && c.key == key) { b = &c; break; }
  if (!b)
    for (WarpBar& c : wp.bars)
      if (!c.arrived) { b = &c; break; }
  if (!b) {
    fprintf(stderr, "polykernel: internal error: warp barrier table full\n");
    abort();
  }
  b->key = key;
  unsigned g = b->gen;
  b->arrived |= bit;
  ++w->progress;
  if ((b->arrived & wp.live) == (key & wp.live)) {
    b->arrived = 0;
    ++b->gen;
    return;
  }
  Fiber* f = w->cur;
  wp.waiting |= bit;
  do yield_to_sched(w, f);
  while (b->gen == g);
  wp.waiting &= ~bit;
}

// Lanes that reach this call site together. A lane joins the rendezvous for `site`; the
// group is settled once every other live lane of the warp has joined, is waiting at some
// barrier or at another rendezvous, or has exited - i.e. it cannot join any more.
unsigned converged_mask(uintptr_t site) {
  Worker* w = tl_worker;
  if (!w || !w->fiber_mode) return 0xffffffffu;
  Lane* l = tl_lane;
  Warp& wp = w->warps[l->linear >> 5];
  unsigned bit = 1u << (l->linear & 31u);
  Rendezvous* r = nullptr;
  for (Rendezvous& c : wp.rv)
    if (c.pending && c.site == site) { r = &c; break; }
  if (!r)
    for (Rendezvous& c : wp.rv)
      if (!c.pending) { r = &c; break; }
  r->site = site;
  unsigned g = r->gen;
  r->pending |= bit;
  ++w->progress;
  Fiber* f = w->cur;
  for (;;) {
    if (r->gen != g) return r->result;
    unsigned elsewhere = 0;
    for (Rendezvous& c : wp.rv)
      if (&c != r) elsewhere |= c.pending;
    unsigned accounted = r->pending | wp.waiting | elsewhere | ~wp.live;
    if (accounted == 0xffffffffu) {
      r->result = r->pending & wp.live;
      r->pending = 0;
      ++r->gen;
      ++w->progress;
      return r->result;
    }
    yield_to_sched(w, f);
  }
}

__attribute__((noinline)) unsigned converged_here() {
  return converged_mask(reinterpret_cast<uintptr_t>(__builtin_return_address(0)));
}

unsigned active_mask() {
  Worker* w = tl_worker;
  if (!w || !w->fiber_mode) return 0xffffffffu;
  return w->warps[tl_lane->linear >> 5].live;
}

uint64_t* warp_slots() {
  Worker* w = need_fibers("a warp shuffle/vote intrinsic");
  return w->warps[tl_lane->linear >> 5].slots;
}

void grid_sync_unsupported() {
  fprintf(stderr, "polykernel: cooperative_groups::grid_group::sync() is not supported on the CPU back end\n");
  abort();
}

}  // namespace cpu

cudaError_t cpu_launch(const TuInfo* tu, const LaunchCfg& c, void (*thunk)(void*), void* ctx) {
  if (backend() != Backend::Cpu) {
    fprintf(stderr, "polykernel: internal error: CPU launch while the OpenCL back end is active\n");
    abort();
  }
  if (cpu::tl_block) {   // a kernel launching a kernel
    static bool warned = false;
    if (!warned) {
      warned = true;
      fprintf(stderr, "polykernel: launching a kernel from device code (dynamic parallelism) is not supported\n");
    }
    return record_launch_error(cudaErrorNotSupported);
  }
  const dim3 g = c.grid, b = c.block;
  unsigned long long nthreads = (unsigned long long)b.x * b.y * b.z;
  // CUDA 12 reports a block that is too large as cudaErrorInvalidValue (checked on a T4)
  if (nthreads > 1024 || b.z > 64) return record_launch_error(cudaErrorInvalidValue);
  if (g.x == 0 || g.y == 0 || g.z == 0 || b.x == 0 || b.y == 0 || b.z == 0 || g.y > 65535 || g.z > 65535 ||
      g.x > 0x7fffffffu)
    return record_launch_error(cudaErrorInvalidConfiguration);
  cpu::Job J;
  J.thunk = thunk;
  J.ctx = ctx;
  J.grid = g;
  J.block = b;
  J.shmem = c.shmem;
  J.fibers = !tu || tu->needs_fibers || cpu::force_fibers();
  J.nthreads = (unsigned)nthreads;
  J.nblocks = (unsigned long long)g.x * g.y * g.z;
  cpu::Pool::get().run(J);
  return cudaSuccess;
}

}  // namespace pk
