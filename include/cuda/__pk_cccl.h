// PolyKernel: small subset of the CUDA C++ standard library (libcu++/CCCL) for the CPU
// back end: cuda::std::* forwards to the host standard library, plus atomic_ref,
// thread scopes and the cuda/cmath helpers.
#ifndef POLYKERNEL_CCCL_H
#define POLYKERNEL_CCCL_H
#include "../cuda_runtime.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace cuda {
namespace std {
using namespace ::std;

template <class T>
class atomic_ref {
  static_assert(::std::is_trivially_copyable<T>::value, "atomic_ref requires a trivially copyable type");
  T* p_;
  static constexpr int mo(::std::memory_order o) {
    return o == ::std::memory_order_relaxed ? __ATOMIC_RELAXED
         : o == ::std::memory_order_acquire ? __ATOMIC_ACQUIRE
         : o == ::std::memory_order_release ? __ATOMIC_RELEASE
         : o == ::std::memory_order_acq_rel ? __ATOMIC_ACQ_REL
         : o == ::std::memory_order_consume ? __ATOMIC_CONSUME : __ATOMIC_SEQ_CST;
  }
  static constexpr int fail_mo(::std::memory_order o) {
    return (o == ::std::memory_order_acq_rel) ? __ATOMIC_ACQUIRE
         : (o == ::std::memory_order_release) ? __ATOMIC_RELAXED : mo(o);
  }

 public:
  using value_type = T;
  static constexpr bool is_always_lock_free = sizeof(T) <= 8;
  explicit atomic_ref(T& r) : p_(::std::addressof(r)) {}
  atomic_ref(const atomic_ref&) = default;
  T load(::std::memory_order o = ::std::memory_order_seq_cst) const {
    T v;
    __atomic_load(p_, &v, mo(o));
    return v;
  }
  void store(T v, ::std::memory_order o = ::std::memory_order_seq_cst) const { __atomic_store(p_, &v, mo(o)); }
  operator T() const { return load(); }
  T operator=(T v) const { store(v); return v; }
  T exchange(T v, ::std::memory_order o = ::std::memory_order_seq_cst) const {
    T old;
    __atomic_exchange(p_, &v, &old, mo(o));
    return old;
  }
  bool compare_exchange_weak(T& e, T d, ::std::memory_order o = ::std::memory_order_seq_cst) const {
    return __atomic_compare_exchange(p_, &e, &d, true, mo(o), fail_mo(o));
  }
  bool compare_exchange_strong(T& e, T d, ::std::memory_order o = ::std::memory_order_seq_cst) const {
    return __atomic_compare_exchange(p_, &e, &d, false, mo(o), fail_mo(o));
  }
  bool compare_exchange_weak(T& e, T d, ::std::memory_order s, ::std::memory_order f) const {
    return __atomic_compare_exchange(p_, &e, &d, true, mo(s), mo(f));
  }
  bool compare_exchange_strong(T& e, T d, ::std::memory_order s, ::std::memory_order f) const {
    return __atomic_compare_exchange(p_, &e, &d, false, mo(s), mo(f));
  }
  template <class F>
  T update(F f, ::std::memory_order o) const {
    T old = load(::std::memory_order_relaxed);
    while (!compare_exchange_weak(old, f(old), o)) {
    }
    return old;
  }
  T fetch_add(T v, ::std::memory_order o = ::std::memory_order_seq_cst) const {
    if constexpr (::std::is_integral<T>::value) return __atomic_fetch_add(p_, v, mo(o));
    else return update([v](T x) { return x + v; }, o);
  }
  T fetch_sub(T v, ::std::memory_order o = ::std::memory_order_seq_cst) const {
    if constexpr (::std::is_integral<T>::value) return __atomic_fetch_sub(p_, v, mo(o));
    else return update([v](T x) { return x - v; }, o);
  }
  T fetch_and(T v, ::std::memory_order o = ::std::memory_order_seq_cst) const { return __atomic_fetch_and(p_, v, mo(o)); }
  T fetch_or(T v, ::std::memory_order o = ::std::memory_order_seq_cst) const { return __atomic_fetch_or(p_, v, mo(o)); }
  T fetch_xor(T v, ::std::memory_order o = ::std::memory_order_seq_cst) const { return __atomic_fetch_xor(p_, v, mo(o)); }
  T fetch_min(T v, ::std::memory_order o = ::std::memory_order_seq_cst) const {
    return update([v](T x) { return v < x ? v : x; }, o);
  }
  T fetch_max(T v, ::std::memory_order o = ::std::memory_order_seq_cst) const {
    return update([v](T x) { return v > x ? v : x; }, o);
  }
  T operator++() const { return fetch_add(T(1)) + T(1); }
  T operator++(int) const { return fetch_add(T(1)); }
  T operator--() const { return fetch_sub(T(1)) - T(1); }
  T operator--(int) const { return fetch_sub(T(1)); }
  T operator+=(T v) const { return fetch_add(v) + v; }
  T operator-=(T v) const { return fetch_sub(v) - v; }
  bool is_lock_free() const { return sizeof(T) <= 8; }
};
}  // namespace std

enum thread_scope { thread_scope_system = 0, thread_scope_device = 1, thread_scope_block = 2, thread_scope_thread = 3 };

template <class T, thread_scope Scope = thread_scope_system>
class atomic_ref : public std::atomic_ref<T> {
 public:
  using std::atomic_ref<T>::atomic_ref;
  using std::atomic_ref<T>::operator=;
};

template <class T, thread_scope Scope = thread_scope_system>
class atomic : public ::std::atomic<T> {
 public:
  using ::std::atomic<T>::atomic;
  using ::std::atomic<T>::operator=;
  T fetch_min(T v, ::std::memory_order o = ::std::memory_order_seq_cst) {
    T old = this->load(::std::memory_order_relaxed);
    while (v < old && !this->compare_exchange_weak(old, v, o)) {
    }
    return old;
  }
  T fetch_max(T v, ::std::memory_order o = ::std::memory_order_seq_cst) {
    T old = this->load(::std::memory_order_relaxed);
    while (v > old && !this->compare_exchange_weak(old, v, o)) {
    }
    return old;
  }
};

inline void atomic_thread_fence(::std::memory_order o, thread_scope = thread_scope_system) {
  ::std::atomic_thread_fence(o);
}

template <class T, class U>
constexpr typename ::std::common_type<T, U>::type ceil_div(T a, U b) {
  typedef typename ::std::common_type<T, U>::type C;
  return (C)(((C)a + (C)b - 1) / (C)b);
}
template <class T, class U>
constexpr typename ::std::common_type<T, U>::type round_up(T a, U b) {
  return ceil_div(a, b) * b;
}
template <class T, class U>
constexpr typename ::std::common_type<T, U>::type round_down(T a, U b) {
  typedef typename ::std::common_type<T, U>::type C;
  return ((C)a / (C)b) * (C)b;
}
}  // namespace cuda
#endif
