// Tiled matrix multiply: 2D grid/blocks, static __shared__ arrays, __syncthreads.
#include "common.h"
#define TILE 16

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int ak = t * TILE + threadIdx.x;
    int bk = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (col < N && bk < K) ? B[bk * N + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

int main() {
  const int M = 130, N = 67, K = 99;
  float *hA = new float[M * K], *hB = new float[K * N], *hC = new float[M * N];
  for (int i = 0; i < M * K; ++i) hA[i] = (float)((i * 7) % 13) - 6.0f;
  for (int i = 0; i < K * N; ++i) hB[i] = (float)((i * 5) % 11) - 5.0f;
  float *dA, *dB, *dC;
  CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
  CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
  CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
  CHECK(cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice));
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  matmul<<<grid, block>>>(dA, dB, dC, M, N, K);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c) {
      float want = 0.0f;
      for (int k = 0; k < K; ++k) want += hA[r * K + k] * hB[k * N + c];
      EXPECT(fabsf(hC[r * N + c] - want) < 1e-3f, "C[%d][%d] = %f, want %f", r, c, hC[r * N + c], want);
    }
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  delete[] hA; delete[] hB; delete[] hC;
  printf("PASS t02_matmul_tiled\n");
  return 0;
}
