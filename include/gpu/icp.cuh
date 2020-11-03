#pragma once

#include <vector>
#include <limits>
#include <tuple>
#include <iostream>
#include <cmath>

#include "libalg/basic_operations.hpp"
#include "libalg/alg.hpp"
#include "libalg/CPUMatrix.hpp"
#include "cpu/icp.hpp"
#include "libalg/CPUView.hpp"
#include "error.hpp"

#define UNUSED(x) (void)x

/* --------- CPU FUNCTIONS ---------*/

__host__ std::vector<std::tuple<size_t, int>> get_correspondence_indices(float *P, float *Q, size_t P_r, size_t P_c, size_t Q_r, size_t Q_c);
__host__ void increment_cov(float *P, float *Q);
__host__ void compute_cross_variance_array(float * cov, float *P, float *Q, std::tuple<size_t, int> *correspondences, size_t P_r, size_t P_c,
                                size_t Q_r, size_t Q_c);

/* ------------ Calling kernel functions -------- */
__host__ float* calling_transpose_kernel(float *A, size_t row, size_t column);
__host__ float *calling_dot_kernel(float *A, float *B, size_t A_row, size_t A_col, size_t B_row, size_t B_col);

/* --------- GPU FUNCTIONS ---------*/
__global__ void get_correspondence_indices_array_gpu(float *P, float *Q, size_t P_r, size_t P_c, size_t Q_r, size_t Q_c, std::tuple<size_t, int> *correspondances);

void compute_cross_variance_array(float * cov, float *P, float *Q, std::tuple<size_t, int> *correspondences, size_t P_r, size_t P_c,
                                size_t Q_r, size_t Q_c);

void gpuTranspose(float *A, float *B, int numRows, int numColumns);
__global__ void naiveGPUTranspose(const float *d_a, float *d_b, const int rows, const int cols);

CPUMatrix icp_gpu(CPUMatrix& P, CPUMatrix& Q, unsigned iterations);
CPUMatrix icp_gpu_optimized(CPUMatrix& P, CPUMatrix& Q, unsigned iterations);
