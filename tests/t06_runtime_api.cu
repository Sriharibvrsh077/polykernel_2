// Runtime API surface: device query, streams, events, async copies, pinned and managed
// memory, 2D copies, error reporting, cudaLaunchKernel, pointer attributes.
#include "common.h"
#include <cstring>

__global__ void fill(int* p, int n, int v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v + i;
}
__global__ void increment(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 1.0f;
}
__global__ void scale2d(float* p, size_t pitch, int w, int h, float s) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    float* row = (float*)((char*)p + y * pitch);
    row[x] *= s;
  }
}

int main() {
  int count = 0;
  CHECK(cudaGetDeviceCount(&count));
  EXPECT(count >= 1, "device count %d", count);
  CHECK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  EXPECT(prop.warpSize == 32 && prop.maxThreadsPerBlock >= 256 && strlen(prop.name) > 0, "properties");
  int sms = 0, major = 0;
  CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
  EXPECT(sms >= 1 && major >= 1, "attributes");
  int rt = 0;
  CHECK(cudaRuntimeGetVersion(&rt));
  printf("device: %s, %d SMs, cc %d.%d, runtime %d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor, rt);

  // streams + events + async copies with pinned host memory
  const int n = 1 << 16;
  int* h = nullptr;
  CHECK(cudaMallocHost(&h, n * sizeof(int)));
  int* d = nullptr;
  CHECK(cudaMalloc(&d, n * sizeof(int)));
  cudaStream_t s1, s2;
  CHECK(cudaStreamCreate(&s1));
  CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start, s1));
  fill<<<n / 256, 256, 0, s1>>>(d, n, 7);
  CHECK(cudaGetLastError());
  CHECK(cudaEventRecord(stop, s1));
  CHECK(cudaEventSynchronize(stop));
  float ms = -1.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  EXPECT(ms >= 0.0f, "elapsed time %f", ms);
  CHECK(cudaMemcpyAsync(h, d, n * sizeof(int), cudaMemcpyDeviceToHost, s2));
  CHECK(cudaStreamSynchronize(s2));
  for (int i = 0; i < n; ++i) EXPECT(h[i] == 7 + i, "fill[%d] = %d", i, h[i]);

  // device-to-device copy and memset
  int* d2;
  CHECK(cudaMalloc(&d2, n * sizeof(int)));
  CHECK(cudaMemcpy(d2, d, n * sizeof(int), cudaMemcpyDeviceToDevice));
  CHECK(cudaMemset(d2, 0, 16 * sizeof(int)));
  CHECK(cudaMemcpy(h, d2, n * sizeof(int), cudaMemcpyDefault));
  for (int i = 0; i < n; ++i) EXPECT(h[i] == (i < 16 ? 0 : 7 + i), "d2d[%d] = %d", i, h[i]);

  // managed memory: host writes, kernel updates, host reads after synchronize
  float* m = nullptr;
  CHECK(cudaMallocManaged(&m, 1000 * sizeof(float)));
  for (int i = 0; i < 1000; ++i) m[i] = (float)i;
  increment<<<4, 256>>>(m, 1000);
  increment<<<4, 256>>>(m, 1000);
  CHECK(cudaDeviceSynchronize());
  for (int i = 0; i < 1000; ++i) EXPECT(m[i] == (float)i + 2.0f, "managed[%d] = %f", i, m[i]);
  cudaPointerAttributes attr;
  CHECK(cudaPointerGetAttributes(&attr, d));
  EXPECT(attr.type == cudaMemoryTypeDevice, "pointer attributes type %d", (int)attr.type);

  // pitched 2D memory
  const int w = 37, hgt = 11;
  size_t pitch = 0;
  float* p2;
  CHECK(cudaMallocPitch(&p2, &pitch, w * sizeof(float), hgt));
  EXPECT(pitch >= w * sizeof(float), "pitch");
  float host2d[hgt][w];
  for (int y = 0; y < hgt; ++y)
    for (int x = 0; x < w; ++x) host2d[y][x] = (float)(y * 100 + x);
  CHECK(cudaMemcpy2D(p2, pitch, host2d, w * sizeof(float), w * sizeof(float), hgt, cudaMemcpyHostToDevice));
  scale2d<<<dim3((w + 15) / 16, (hgt + 7) / 8), dim3(16, 8)>>>(p2, pitch, w, hgt, 0.5f);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy2D(host2d, w * sizeof(float), p2, pitch, w * sizeof(float), hgt, cudaMemcpyDeviceToHost));
  for (int y = 0; y < hgt; ++y)
    for (int x = 0; x < w; ++x) EXPECT(host2d[y][x] == 0.5f * (y * 100 + x), "2d[%d][%d]", y, x);

  // cudaLaunchKernel with an argument array
  int nn = 256, base = 100;
  void* args[] = {&d, &nn, &base};
  CHECK(cudaLaunchKernel((const void*)fill, dim3(1), dim3(256), args, 0, 0));
  CHECK(cudaMemcpy(h, d, 256 * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 256; ++i) EXPECT(h[i] == 100 + i, "cudaLaunchKernel[%d] = %d", i, h[i]);

  // error reporting: invalid launch configuration
  fill<<<1, 4096>>>(d, n, 0);
  cudaError_t e = cudaGetLastError();
  // CUDA 12 returns cudaErrorInvalidValue for an oversized block; older releases returned
  // cudaErrorInvalidConfiguration
  EXPECT(e == cudaErrorInvalidValue || e == cudaErrorInvalidConfiguration, "expected a launch error, got %s",
         cudaGetErrorName(e));
  EXPECT(cudaGetLastError() == cudaSuccess, "error not cleared");
  EXPECT(cudaFree(nullptr) == cudaSuccess, "cudaFree(nullptr)");
  EXPECT(strlen(cudaGetErrorString(cudaErrorMemoryAllocation)) > 0, "error string");

  size_t fr = 0, tot = 0;
  CHECK(cudaMemGetInfo(&fr, &tot));
  EXPECT(tot > 0, "mem info");
  CHECK(cudaStreamDestroy(s1));
  CHECK(cudaStreamDestroy(s2));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  CHECK(cudaFree(d));
  CHECK(cudaFree(d2));
  CHECK(cudaFree(m));
  CHECK(cudaFree(p2));
  CHECK(cudaFreeHost(h));
  CHECK(cudaDeviceReset());
  printf("PASS t06_runtime_api\n");
  return 0;
}
