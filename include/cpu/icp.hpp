#pragma once

#include <vector>
#include <tuple>

#include "libalg/alg.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/CPUView.hpp"
#include "cpu/tuple.hpp"

std::vector<std::tuple<size_t, int>> get_correspondence_indices(CPUMatrix &P, CPUMatrix &Q);
std::vector<std::tuple<size_t, int>> get_correspondence_indices(float *P, float *Q, size_t P_r,
                                                                 size_t P_c, size_t Q_r, size_t Q_c);
void get_correspondence_indices_array(tuple **correspondances, float *P, float *Q, size_t P_r, size_t P_c, size_t Q_r, size_t Q_c);

float default_kernel(CPUMatrix a);
float default_kernel(float a);
void increment_cov(float *P, float *Q);

std::tuple<CPUMatrix, std::vector<float>> compute_cross_variance(CPUMatrix &P, CPUMatrix &Q,
    const std::vector<std::tuple<size_t, int>>& correspondences, float (*kernel)(CPUMatrix a));

float *compute_cross_variance_cpu_call_gpu(float *P, float *Q, std::vector<std::tuple<size_t, int>> correspondences, size_t P_r, size_t P_c,
                                size_t Q_r, size_t Q_c);

void compute_cross_variance_array(float * cov, float *P, float *Q, std::tuple<size_t, int> *correspondences, size_t P_r, size_t P_c,
                                size_t Q_r, size_t Q_c);

std::tuple<CPUMatrix, std::vector<float>, std::vector<std::tuple<size_t, int>>> icp(CPUMatrix &P, CPUMatrix &Q, unsigned iterations);