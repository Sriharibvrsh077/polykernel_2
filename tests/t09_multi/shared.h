#pragma once
__global__ void axpb(float* y, const float* x, float a, float b, int n);
void launch_axpb(float* y, const float* x, float a, float b, int n);
