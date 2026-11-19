// ... and launched from another (extern kernel declaration in a shared header).
#include "../common.h"
#include "shared.h"
int main() {
  const int n = 5000;
  float hx[n], hy[n];
  for (int i = 0; i < n; ++i) hx[i] = (float)i;
  float *dx, *dy;
  CHECK(cudaMalloc(&dx, sizeof hx));
  CHECK(cudaMalloc(&dy, sizeof hy));
  CHECK(cudaMemcpy(dx, hx, sizeof hx, cudaMemcpyHostToDevice));
  axpb<<<(n + 255) / 256, 256>>>(dy, dx, 2.0f, 1.0f, n);   // launch from this TU
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(hy, dy, sizeof hy, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) EXPECT(hy[i] == 2.0f * i + 1.0f, "axpb[%d]", i);
  launch_axpb(dy, dx, -1.0f, 0.5f, n);                     // launch from the defining TU
  CHECK(cudaMemcpy(hy, dy, sizeof hy, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) EXPECT(hy[i] == -1.0f * i + 0.5f, "launch_axpb[%d]", i);
  printf("PASS t09_multi\n");
  return 0;
}
