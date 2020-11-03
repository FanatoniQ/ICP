#pragma once

#define is_power_of_2(x) (x & (x-1)) == 0

__global__ void squared_norm_2_kernel(float *d_A, float *d_B, float *d_res, int pitch, int width, int height, int reducepitch);

__host__ float cuda_squared_norm_2(float *d_A, float *d_B, size_t width, size_t height, size_t pitch, int threads);

__global__ void squared_norm_2_kernel_0(const float *d_A, const float *d_B, float *d_res, int pitch, int width, int height, int reducepitch);

__host__ float cuda_squared_norm_2_0(float *d_A, float *d_B, size_t width, size_t height, size_t pitch, int threads);
