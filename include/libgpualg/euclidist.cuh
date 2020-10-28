#pragma once

#define is_power_of_2(x) (x & (x-1)) == 0

__global__ void euclidist_kernel(double *d_A, double *d_B, double *d_res, int pitch, int width, int height, int reducepitch);
