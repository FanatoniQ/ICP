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



__global__ void get_correspondence_indices(CPUMatrix &P, CPUMatrix &Q, std::vector<std::tuple<size_t, int>> &correspondances);

__global__ void compute_cross_variance(CPUMatrix &P, CPUMatrix &Q, const std::vector<std::tuple<size_t, int>> &correspondences,
                            double (*kernel)(CPUMatrix a), std::tuple<CPUMatrix, std::vector<double>> &res);

__global__ void naiveGPUTranspose(const double *d_a, double *d_b, const int rows, const int cols);