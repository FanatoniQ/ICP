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

#include "gpu/icp.cuh"

#define UNUSED(x) (void)x

/* --------- CPU Version Calling GPU Kernel ------------ */
std::vector<std::tuple<size_t, int>> get_correspondence_indices(double *P, double *Q,
                                                                size_t P_r, size_t P_c, size_t Q_r, size_t Q_c)
{
    std::vector<std::tuple<size_t, int>> correspondances = {};
    for (size_t i = 0; i < P_r; i++)
    {
        double *p_point = P + i * P_c;
        double min_dist = std::numeric_limits<double>::max();
        int chosen_idx = -1;
        for (size_t j = 0; j < Q_r; j++)
        {
            double *q_point = Q + j * Q_c;
            double dist = std::sqrt(element_wise_reduce(p_point, q_point, 1, P_c, 1, Q_c,
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


// Intermediation function to be replaced with element_wise_op
void increment_cov(double *P, double *Q)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            P[i*3 + j] = P[i*3 + j] + Q[i*3 + j];
        }
    }
}

double *compute_cross_variance_cpu_call_gpu(double *P, double *Q, std::vector<std::tuple<size_t, int>> correspondences, size_t P_r, size_t P_c,
                                size_t Q_r, size_t Q_c) //set default function to lambda function??
{
    UNUSED(Q_r);
    double *cov = (double *)calloc(9, sizeof(double));

    for (auto tup : correspondences)
    {
        auto i = std::get<0>(tup);
        auto j = std::get<1>(tup);
        double *q_point = Q + j * Q_c;
        double *p_point = P + i * P_c;

        double *doted_points = nullptr;
        double *transposed_Q = transpose(q_point, 1, Q_c);
        dot_product(&doted_points, transposed_Q, p_point, Q_c, 1, 1, P_c); //dim of Q_r * P_r
        free (transposed_Q); 
        increment_cov(cov, doted_points); //need to set element_wise_op but too complicated, doesn't work for some reason.
        free(doted_points);
    }
    return cov;
}

/* -------------- Version GPU Kernel -----------*/

// Implementation with double arrays and no vector for full GPU usage
void get_correspondence_indices_array(double *P, double *Q,
                                size_t P_r, size_t P_c, size_t Q_r, size_t Q_c, std::tuple<size_t, int> *correspondances)
{
    int push_index = 0;
    for (size_t i = 0; i < P_r; i++)
    {
        double *p_point = P + i * P_c;
        double min_dist = std::numeric_limits<double>::max();
        int chosen_idx = -1;
        for (size_t j = 0; j < Q_r; j++)
        {
            double *q_point = Q + j * Q_c;
            double dist = std::sqrt(element_wise_reduce(p_point, q_point, 1, P_c, 1, Q_c,
                                    squared_norm_2, add, add)); //norm 2 between 2 vectors
            if (dist < min_dist)
            {
                min_dist = dist;
                chosen_idx = j;
            }
        }
        correspondances[push_index] = std::make_tuple(i, chosen_idx);
        push_index++;
    }
}

// Array implementation for GPU
void compute_cross_variance_array(double * cov, double *P, double *Q, std::tuple<size_t, int> *correspondences, size_t P_r, size_t P_c,
                                size_t Q_r, size_t Q_c) //set default function to lambda function??
{
    UNUSED(Q_r);
    UNUSED(P_r);

    for (size_t index = 0; index < P_r; index ++)
    {
        auto i = std::get<0>(correspondences[index]);
        auto j = std::get<1>(correspondences[index]);
        double *q_point = Q + j * Q_c;
        double *p_point = P + i * P_c;

        double *doted_points = nullptr;
        double *transposed_Q = transpose(q_point, 1, Q_c);
        dot_product(&doted_points, transposed_Q, p_point, Q_c, 1, 1, P_c); //dim of Q_r * P_r
        free (transposed_Q); 
        increment_cov(cov, doted_points); //need to set element_wise_op but too complicated, doesn't work for some reason.
        free(doted_points);
    }
}

__global__ void naiveGPUTranspose(const double *d_a, double *d_b, const int rows, const int cols) 
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int index_in = i * cols + j;
    int index_out = j * rows + i;

    if (i < rows && j < cols)
        d_b[index_out] = d_a[index_in];
}