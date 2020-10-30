#pragma once

#define is_power_of_2(x) (x & (x-1)) == 0

__global__ void squared_norm_2_kernel(double *d_A, double *d_B, double *d_res, int pitch, int width, int height, int reducepitch);

__host__ double cuda_squared_norm_2(double *d_A, double *d_B, size_t width, size_t height, size_t pitch, int threads);

__global__ void squared_norm_2_kernel_0(const double *d_A, const double *d_B, double *d_res, int pitch, int width, int height, int reducepitch);

__host__ double cuda_squared_norm_2_0(double *d_A, double *d_B, size_t width, size_t height, size_t pitch, int threads);
