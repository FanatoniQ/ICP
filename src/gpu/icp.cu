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

// Version CPU calling GPU kernel

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
void increment_array(double *P, double *Q)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            P[i*3 + j] = P[i*3 + j] + Q[i*3 + j];
        }
    }
}

// Need to rewrite it here otherwise not recognized from cpp file
double default_kernel(double a)
{
    UNUSED(a);
    return 1;
}

// Array implementation for GPU
std::tuple<double *, std::vector<double>> compute_cross_variance(double *P, double *Q,
                                                                 std::vector<std::tuple<size_t, int>> correspondences, size_t P_r, size_t P_c,
                                                                 size_t Q_r, size_t Q_c, double (*kernel)(double a)) //set default function to lambda function??
{
    UNUSED(Q_r);
    UNUSED(P_r);
    if (kernel == nullptr)
        kernel = &default_kernel;
    double *cov = (double *)calloc(9, sizeof(double));
    std::vector<double> exclude_indices = {};
    for (auto tup : correspondences)
    {
        auto i = std::get<0>(tup);
        auto j = std::get<1>(tup);
        double *q_point = Q + j * Q_c;
        double *p_point = P + i * P_c;
        double weight = kernel(*p_point - *q_point);

        if (weight < 0.01)
            exclude_indices.push_back(i);

        double *doted_points = nullptr;
        dot_product(&doted_points, transpose(q_point, 1, Q_c), p_point, Q_c, 1, 1, P_c); //dim of Q_r * P_r

        double *weighted_points = nullptr; //multiply by the weight
        size_t Wdim0, Wdim1;                // use high level API instead
        element_wise_op(&weighted_points, &weight, doted_points, 1, 1, Q_c, P_c, Wdim0, Wdim1, mult);
        //size_t Wcov0, Wcov1; // should be 2, use high level API instead
        //element_wise_op(&cov, cov, weighted_points, 3, 3, P_c, Q_c, Wcov0, Wcov1, add);
        increment_array(cov, weighted_points); //need to set element_wise_op but too complicated, doesn't work for some reason.
        free(weighted_points);
        free(doted_points);
    }
    return std::make_tuple(cov, exclude_indices);
}

// Version GPU Kernel
/*
void get_correspondence_indices(std::vector<std::tuple<size_t, int>> res, double *P, double *Q, size_t P_r, size_t P_c, size_t Q_r, size_t Q_c)
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
*/