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

// Implementation with float arrays, calling kernels
std::vector<std::tuple<size_t, int>> get_correspondence_indices(float *P, float *Q,
                                                                size_t P_r, size_t P_c, size_t Q_r, size_t Q_c)
{
    std::vector<std::tuple<size_t, int>> correspondances = {};
    for (size_t i = 0; i < P_r; i++)
    {
        float *p_point = P + i * P_c;
        float min_dist = std::numeric_limits<float>::max();
        int chosen_idx = -1;
        for (size_t j = 0; j < Q_r; j++)
        {
            float *q_point = Q + j * Q_c;
            float dist = std::sqrt(element_wise_reduce(p_point, q_point, 1, P_c, 1, Q_c,
                                    squared_norm_2, add, add)); //norm 2 between 2 vectors
            if (dist < min_dist)
            {
                min_dist = dist;
                chosen_idx = j;
            }
        }
        correspondances.push_back(std::make_tuple(i, chosen_idx));
    }
    return correspondances;
}

// Implementation with float arrays and no vector for full GPU usage
void get_correspondence_indices_array(tuple **correspondances, float *P, float *Q, size_t P_r, size_t P_c, size_t Q_r,
                                        size_t Q_c)
{
    int push_index = 0;
    for (size_t i = 0; i < P_r; i++)
    {
        float *p_point = P + i * P_c;
        float min_dist = std::numeric_limits<float>::max();
        int chosen_idx = -1;
        for (size_t j = 0; j < Q_r; j++)
        {
            float *q_point = Q + j * Q_c;
            float dist = std::sqrt(element_wise_reduce(p_point, q_point, 1, P_c, 1, Q_c,
                                    squared_norm_2, add, add)); //norm 2 between 2 vectors
            if (dist < min_dist)
            {
                min_dist = dist;
                chosen_idx = j;
            }
        }
        tuple *new_tup = (tuple*)calloc(1, sizeof(tuple));
        new_tup->index = i;
        new_tup->value = chosen_idx;
        correspondances[push_index] = new_tup;
        push_index++;
    }
}

// Implementation with CPUMatrix for quick CPU usage
std::vector<std::tuple<size_t, int>> get_correspondence_indices(CPUMatrix &P, CPUMatrix &Q)
{
    std::vector<std::tuple<size_t, int>> correspondances = {};
    for (size_t i = 0; i < P.getDim0(); i++)
    {
        auto p_point = P.getLine(i);
        float min_dist = std::numeric_limits<float>::max();
        int chosen_idx = -1;
        for (size_t j = 0; j < Q.getDim0(); j++)
        {
            auto q_point = Q.getLine(j);
            float dist = std::sqrt(p_point.euclidianDistance(q_point));
            if (dist < min_dist)
            {
                min_dist = dist;
                chosen_idx = j;
            }
        }
        correspondances.push_back(std::make_tuple(i, chosen_idx));
    }
    return correspondances;
}

float default_kernel(CPUMatrix a)
{
    UNUSED(a);
    return 1;
}

float default_kernel(float a)
{
    UNUSED(a);
    return 1;
}

// Implementation with CPUMAtrix
std::tuple<CPUMatrix, std::vector<float>> compute_cross_variance(CPUMatrix &P, CPUMatrix &Q,
                                                                  const std::vector<std::tuple<size_t, int>> &correspondences, float (*kernel)(CPUMatrix a))
{
    if (kernel == nullptr)
        kernel = &default_kernel;
    CPUMatrix cov = CPUMatrix(P.getDim1(), P.getDim1());
    std::vector<float> exclude_indices = {};
    for (auto tup : correspondences)
    {
        auto i = std::get<0>(tup);
        auto j = std::get<1>(tup);
        CPUView q_point = Q.getLine(j);
        CPUView p_point = P.getLine(i);
        float weight = kernel(p_point - q_point);

        if (weight < 0.01)
            exclude_indices.push_back(i);

        CPUMatrix doted_points = q_point.transpose().dot(p_point);
        doted_points *= weight;
        cov += doted_points;
    }
    return std::make_tuple(std::move(cov), exclude_indices);
}

// Intermediation function to be replaced with element_wise_op
// We know this is only supposed to work with 3 dimensions so cov is 3*3
void increment_cov(float *P, float *Q)
{
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            P[i*3 + j] = P[i*3 + j] + Q[i*3 + j];
        }
    }
}

// Array implementation for GPU
float *compute_cross_variance_cpu_call_gpu(float *P, float *Q, std::vector<std::tuple<size_t, int>> correspondences, size_t P_r, size_t P_c,
                                size_t Q_r, size_t Q_c) //set default function to lambda function??
{
    UNUSED(Q_r);
    UNUSED(P_r);
    float *cov = (float *)calloc(9, sizeof(float));

    for (auto tup : correspondences)
    {
        auto i = std::get<0>(tup);
        auto j = std::get<1>(tup);
        float *q_point = Q + j * Q_c;
        float *p_point = P + i * P_c;

        float *doted_points = nullptr;
        float *transposed_Q = transpose(q_point, 1, Q_c);
        dot_product(&doted_points, transposed_Q, p_point, Q_c, 1, 1, P_c); //dim of Q_r * P_r
        free (transposed_Q); 
        increment_cov(cov, doted_points); //need to set element_wise_op but too complicated, doesn't work for some reason.
        free(doted_points);
    }
    return cov;
}

// Array implementation for GPU
void compute_cross_variance_array(float * cov, float *P, float *Q, std::tuple<size_t, int> *correspondences, size_t P_r, size_t P_c,
                                size_t Q_r, size_t Q_c) //set default function to lambda function??
{
    UNUSED(Q_r);

    for (size_t index = 0; index < P_r; index ++)
    {
        auto i = std::get<0>(correspondences[index]);
        auto j = std::get<1>(correspondences[index]);
        float *q_point = Q + j * Q_c;
        float *p_point = P + i * P_c;

        float *doted_points = nullptr;
        float *transposed_Q = transpose(q_point, 1, Q_c);
        dot_product(&doted_points, transposed_Q, p_point, Q_c, 1, 1, P_c); //dim of Q_r * P_r
        free (transposed_Q); 
        increment_cov(cov, doted_points); //need to set element_wise_op but too complicated, doesn't work for some reason.
        free(doted_points);
    }
}

// Quick CPUMatrix implementation for CPU
std::tuple<CPUMatrix, std::vector<float>, std::vector<std::tuple<size_t, int>>> icp(CPUMatrix &P, CPUMatrix &Q, unsigned iterations)
{
    // Center data P and Q
    auto Q_center = Q.mean(0);
    Q -= Q_center;

    std::vector<std::tuple<size_t, int>> correps_values;
    std::vector<float> norm_values;
    CPUMatrix P_copy;
    P_copy = P;
    for (unsigned i = 0; i < iterations; ++i)
    {
        auto P_center = P_copy.mean(0);
        // Center P
        P = P_copy - P_center;
        // Compute correspondences indices
        auto corresps = get_correspondence_indices(P, Q);

        correps_values.insert(correps_values.end(), corresps.begin(), corresps.end());
        norm_values.push_back(P.euclidianDistance(Q));
        auto cross_var = compute_cross_variance(P, Q, corresps, default_kernel);
        auto cross_var_other = P.transpose().dot(Q);
        // cross_var is here 3*3 mat
        // U, S, V_T = svd
        auto [U, S, V_T] = std::get<0>(cross_var).svd();
        std::cout << "U: \n"
                  << U << std::endl;
        std::cout << "S: \n"
                  << S << std::endl;
        std::cout << "V_T: \n"
                  << V_T << std::endl;
        UNUSED(S); // unused
        // Rotation matrix
        auto R = U.dot(V_T);
        // Translation Matrix
        auto t = Q_center - P_center.dot(R.transpose());
        // Update P
        P_copy = P_copy.dot(R.transpose()) + t;
    }
    correps_values.push_back(correps_values.back());
    return std::make_tuple(std::move(P_copy), norm_values, correps_values);
}


// \deprecated use CPUMatrix::euclidianDistance instead
/**
float three_dim_norm(CPUMatrix A)
{
    //if (A.getDim1() != 3)
    //    throw std::invalid_argument("Matrix not of dim 3");
    float r = 0;
    for (size_t i = 0; i < A.getDim1(); ++i)
        r += pow2(A(0, i));
    if (A.getDim1() == 3)
        runtime_assert((std::pow(A(0, 0), 2) + std::pow(A(0, 1), 2) + std::pow(A(0, 2), 2)) == r, "FATAL");
    auto norm = A.squared_norm(-1);
    runtime_assert(norm.getDim0() == 1 && norm.getDim1() == 1, "INVALID NORM SIZE ! FATAL ERROR");
    float res = norm(0, 0);
    runtime_assert(r == res, "INVALID NORM ! FATAL ERROR");
    //return std::pow(A(0, 0), 2) + std::pow(A(0, 1), 2) + std::pow(A(0, 2), 2);
    return r;
}**/