// Kernels defined in one translation unit ...
#include "../common.h"
#include "shared.h"
__global__ void axpb(float* y, const float* x, float a, float b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + b;
}
void launch_axpb(float* y, const float* x, float a, float b, int n) {
  axpb<<<(n + 127) / 128, 128>>>(y, x, a, b, n);
}
