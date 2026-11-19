// __constant__ memory, __device__ globals, struct and vector-type arguments,
// __device__ helpers, 3D grids/blocks, C++ casts, file-scope constants, double math.
#include "common.h"

const int NCOEF = 8;
static const float kScale = 0.5f;
__constant__ float coeffs[NCOEF];
__device__ int g_hits;
__device__ unsigned int g_calls = 0;

struct Params {
  float scale;
  int offset;
  float3 bias;
  double weight;
};

enum Mode { MODE_ADD = 1, MODE_MUL = 2 };

__device__ __forceinline__ float poly(float x) {
  float r = 0.0f;
  for (int i = NCOEF - 1; i >= 0; --i) r = r * x + coeffs[i];
  return r;
}

__host__ __device__ inline float clampf(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }

__global__ void evaluate(float* out, const float* in, int nx, int ny, int nz, Params p, Mode mode, bool flag) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= nx || y >= ny || z >= nz) return;
  int idx = (z * ny + y) * nx + x;
  float v = poly(in[idx]) * p.scale + float(p.offset) + p.bias.x + p.bias.y * static_cast<float>(p.bias.z);
  if (mode == MODE_MUL) v *= 2.0f;
  if (flag) v = clampf(v, -100.0f, 100.0f);
  v += (float)(p.weight * sqrt((double)idx)) * kScale;
  out[idx] = v;
  if (v > 0.0f) atomicAdd(&g_hits, 1);
  atomicAdd(&g_calls, 1u);
}

__global__ void vec3_scale(float3* v, int n, float s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float3 a = v[i];
    v[i] = make_float3(a.x * s, a.y * s, a.z * s);
  }
}

static float host_poly(const float* c, float x) {
  float r = 0.0f;
  for (int i = NCOEF - 1; i >= 0; --i) r = r * x + c[i];
  return r;
}

int main() {
  const int nx = 19, ny = 7, nz = 5, n = nx * ny * nz;
  float hc[NCOEF];
  for (int i = 0; i < NCOEF; ++i) hc[i] = 0.1f * (i + 1) * (i % 2 ? -1.0f : 1.0f);
  CHECK(cudaMemcpyToSymbol(coeffs, hc, sizeof(hc)));
  float back[NCOEF];
  CHECK(cudaMemcpyFromSymbol(back, coeffs, sizeof(back)));
  for (int i = 0; i < NCOEF; ++i) EXPECT(back[i] == hc[i], "constant readback %d", i);
  int zero = 0;
  CHECK(cudaMemcpyToSymbol(g_hits, &zero, sizeof(int)));

  float* hin = new float[n];
  for (int i = 0; i < n; ++i) hin[i] = (float)(i % 23) / 23.0f - 0.3f;
  float *din, *dout;
  CHECK(cudaMalloc(&din, n * sizeof(float)));
  CHECK(cudaMalloc(&dout, n * sizeof(float)));
  CHECK(cudaMemcpy(din, hin, n * sizeof(float), cudaMemcpyHostToDevice));
  Params p;
  p.scale = 3.0f;
  p.offset = -1;
  p.bias = make_float3(0.25f, 2.0f, 0.5f);
  p.weight = 0.01;
  dim3 block(4, 4, 2);
  dim3 grid((nx + 3) / 4, (ny + 3) / 4, (nz + 1) / 2);
  evaluate<<<grid, block>>>(dout, din, nx, ny, nz, p, MODE_MUL, true);
  CHECK(cudaGetLastError());
  float* hout = new float[n];
  CHECK(cudaMemcpy(hout, dout, n * sizeof(float), cudaMemcpyDeviceToHost));
  int hits = 0;
  for (int i = 0; i < n; ++i) {
    float v = host_poly(hc, hin[i]) * p.scale + (float)p.offset + p.bias.x + p.bias.y * p.bias.z;
    v *= 2.0f;
    v = fminf(fmaxf(v, -100.0f), 100.0f);
    v += (float)(p.weight * sqrt((double)i)) * kScale;
    EXPECT(fabsf(hout[i] - v) < 1e-4f, "evaluate[%d] = %f, want %f", i, hout[i], v);
    if (v > 0.0f) ++hits;
  }
  int ghits = -1;
  unsigned gcalls = 0;
  CHECK(cudaMemcpyFromSymbol(&ghits, g_hits, sizeof(int)));
  CHECK(cudaMemcpyFromSymbol(&gcalls, g_calls, sizeof(unsigned)));
  EXPECT(ghits == hits, "g_hits %d vs %d", ghits, hits);
  EXPECT(gcalls == (unsigned)n, "g_calls %u vs %d", gcalls, n);

  const int nv = 1000;
  float3* hv = new float3[nv];
  for (int i = 0; i < nv; ++i) hv[i] = make_float3((float)i, (float)-i, 0.5f * i);
  float3* dv;
  CHECK(cudaMalloc(&dv, nv * sizeof(float3)));
  CHECK(cudaMemcpy(dv, hv, nv * sizeof(float3), cudaMemcpyHostToDevice));
  vec3_scale<<<(nv + 63) / 64, 64>>>(dv, nv, 1.5f);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(hv, dv, nv * sizeof(float3), cudaMemcpyDeviceToHost));
  for (int i = 0; i < nv; ++i)
    EXPECT(hv[i].x == 1.5f * i && hv[i].y == -1.5f * i && hv[i].z == 0.75f * i, "vec3_scale %d", i);
  printf("PASS t05_constant_struct\n");
  return 0;
}
