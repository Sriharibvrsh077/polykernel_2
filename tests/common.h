// Shared helpers for the PolyKernel test programs.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define CHECK(x)                                                                          \
  do {                                                                                    \
    cudaError_t e_ = (x);                                                                 \
    if (e_ != cudaSuccess) {                                                              \
      printf("FAIL: %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_));  \
      exit(1);                                                                            \
    }                                                                                     \
  } while (0)
#define EXPECT(cond, ...)              \
  do {                                 \
    if (!(cond)) {                     \
      printf("FAIL: " __VA_ARGS__);    \
      printf("\n");                    \
      exit(1);                         \
    }                                  \
  } while (0)
