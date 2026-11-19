// __device__ / __constant__ variables in every shape the OpenCL back end has to carry as hidden
// buffer arguments: initialised arrays, 2D constant tables, structs, scalars updated by atomics,
// variables reached only through nested device functions, pointers taken from variables,
// cudaGetSymbolAddress and offset copies.
#include "common.h"

struct Config {
  int bias;
  float gain;
  int lut[4];
};

__device__ int g_init[4] = {10, 20, 30, 40};
__constant__ float c_table[3][4];
__constant__ Config c_cfg;
__device__ unsigned int g_counter = 7;
__device__ float g_scratch[64];

__device__ float row_sum(int r) {   // pointer into a __constant__ table
  const float* row = c_table[r];
  float s = 0.f;
  for (int i = 0; i < 4; ++i) s += row[i];
  return s;
}

__device__ int inner(int v) { return v * c_cfg.lut[v & 3] + c_cfg.bias; }   // reached only via outer()

__device__ int outer(int v) {
  atomicAdd(&g_counter, 1u);
  return inner(v) + g_init[v & 3];
}

__global__ void use_vars(int* out_i, float* out_f, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out_i[i] = outer(i);
  out_f[i] = row_sum(i % 3) * c_cfg.gain;
  if (i < 64) g_scratch[i] = (float)i * 0.5f;
}

__global__ void read_through_pointer(const float* p, float* out, int n) {   // p = cudaGetSymbolAddress(g_scratch)
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = p[i] + 1.f;
}

int main() {
  const int n = 100;
  float ht[3][4];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 4; ++c) ht[r][c] = (float)(r * 4 + c);
  CHECK(cudaMemcpyToSymbol(c_table, ht, sizeof(ht)));
  Config cfg = {5, 2.0f, {1, 2, 3, 4}};
  CHECK(cudaMemcpyToSymbol(c_cfg, &cfg, sizeof(cfg)));
  int lut2 = 9;   // overwrite lut[2] with an offset copy
  CHECK(cudaMemcpyToSymbol(c_cfg, &lut2, sizeof(int), offsetof(Config, lut) + 2 * sizeof(int)));
  cfg.lut[2] = 9;

  int *d_i;
  float *d_f, *d_o;
  CHECK(cudaMalloc(&d_i, n * sizeof(int)));
  CHECK(cudaMalloc(&d_f, n * sizeof(float)));
  CHECK(cudaMalloc(&d_o, 64 * sizeof(float)));
  use_vars<<<(n + 31) / 32, 32>>>(d_i, d_f, n);
  CHECK(cudaGetLastError());
  int hi[n];
  float hf[n];
  CHECK(cudaMemcpy(hi, d_i, sizeof(hi), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hf, d_f, sizeof(hf), cudaMemcpyDeviceToHost));
  const int init[4] = {10, 20, 30, 40};
  for (int i = 0; i < n; ++i) {
    int want = i * cfg.lut[i & 3] + cfg.bias + init[i & 3];
    EXPECT(hi[i] == want, "out_i[%d] = %d, want %d", i, hi[i], want);
    int r = i % 3;
    float wf = (ht[r][0] + ht[r][1] + ht[r][2] + ht[r][3]) * cfg.gain;
    EXPECT(fabsf(hf[i] - wf) < 1e-4f, "out_f[%d] = %f, want %f", i, hf[i], wf);
  }
  unsigned counter = 0;
  CHECK(cudaMemcpyFromSymbol(&counter, g_counter, sizeof(counter)));
  EXPECT(counter == 7u + n, "g_counter = %u, want %u", counter, 7u + n);
  int back[2];
  CHECK(cudaMemcpyFromSymbol(back, g_init, sizeof(back), 2 * sizeof(int)));
  EXPECT(back[0] == 30 && back[1] == 40, "g_init offset read: %d %d", back[0], back[1]);
  size_t sz = 0;
  CHECK(cudaGetSymbolSize(&sz, c_table));
  EXPECT(sz == sizeof(ht), "cudaGetSymbolSize = %zu", sz);

  float* p = nullptr;
  CHECK(cudaGetSymbolAddress((void**)&p, g_scratch));
  read_through_pointer<<<2, 32>>>(p, d_o, 64);
  CHECK(cudaGetLastError());
  float ho[64];
  CHECK(cudaMemcpy(ho, d_o, sizeof(ho), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 64; ++i) EXPECT(ho[i] == i * 0.5f + 1.f, "through pointer [%d] = %f", i, ho[i]);
  float hs[4] = {-1, -2, -3, -4};   // cudaMemcpy to a symbol address
  CHECK(cudaMemcpy(p + 10, hs, sizeof(hs), cudaMemcpyHostToDevice));
  float hs2[4];
  CHECK(cudaMemcpyFromSymbol(hs2, g_scratch, sizeof(hs2), 10 * sizeof(float)));
  for (int i = 0; i < 4; ++i) EXPECT(hs2[i] == hs[i], "symbol address write [%d] = %f", i, hs2[i]);

  CHECK(cudaFree(d_i));
  CHECK(cudaFree(d_f));
  CHECK(cudaFree(d_o));
  printf("PASS t12_device_vars\n");
  return 0;
}
