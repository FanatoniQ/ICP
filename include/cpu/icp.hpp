#pragma once

#include <vector>
#include <tuple>

#include "libalg/alg.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/CPUView.hpp"

std::vector<std::tuple<size_t, int>> get_correspondence_indices(double *P,
    double *Q, size_t P_r, size_t P_c, size_t Q_r, size_t Q_c);

double three_dim_norm(CPUMatrix A);

std::vector<std::tuple<size_t, int>> get_correspondence_indices(CPUMatrix &P, CPUMatrix &Q);

double default_kernel(CPUMatrix a);

std::tuple<CPUMatrix, std::vector<double>> compute_cross_variance(CPUMatrix &P, CPUMatrix &Q,
    std::vector<std::tuple<size_t, int>> correspondences, double (*kernel)(CPUMatrix a));

std::tuple<double*, std::vector<double>> compute_cross_variance(double *P, double *Q,
    std::vector<std::tuple<size_t, int>> correspondences, size_t P_r, size_t P_c,
    size_t Q_r, size_t Q_c, double (*kernel)(double a));

std::tuple<std::vector<CPUMatrix>, std::vector<double>, std::vector<std::tuple<size_t, int>>> icp(CPUMatrix P, CPUMatrix Q, unsigned iterations);