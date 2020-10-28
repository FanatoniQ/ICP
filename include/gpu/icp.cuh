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

__host__ std::vector<std::tuple<size_t, int>> get_correspondence_indices(double *P, double *Q, size_t P_r, size_t P_c, size_t Q_r, size_t Q_c);
__host__ void increment_cov(double *P, double *Q);
__host__ void compute_cross_variance_array(double * cov, double *P, double *Q, std::tuple<size_t, int> *correspondences, size_t P_r, size_t P_c,
                                size_t Q_r, size_t Q_c);


/* --------- GPU FUNCTIONS ---------*/
__global__ void get_correspondence_indices_array_gpu(double *P, double *Q, size_t P_r, size_t P_c, size_t Q_r, size_t Q_c, std::tuple<size_t, int> *correspondances);

void compute_cross_variance_array(double * cov, double *P, double *Q, std::tuple<size_t, int> *correspondences, size_t P_r, size_t P_c,
                                size_t Q_r, size_t Q_c);

void gpuTranspose(double *A, double *B, int numRows, int numColumns);
__global__ void naiveGPUTranspose(const double *d_a, double *d_b, const int rows, const int cols);
