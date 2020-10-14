#include <vector>
#include <limits>
#include <tuple>
#include <iostream>

#include "libalg/alg.hpp"
#include "cpu/icp.hpp"

std::vector<std::tuple<size_t, int>> get_correspondence_indices(double *P, double *Q,
    size_t P_r, size_t P_c, size_t Q_r, size_t Q_c)
{
    std::vector<std::tuple<size_t, int>> correspondances = {};
    for (size_t i = 0; i < P_r; i++)
    {
        double *transposed_P = transpose(P, P_r, P_c);
        double *p_point = transposed_P + i*P_c; //begin of line p_point of size P_x
        double min_dist = std::numeric_limits<double>::max();
        int chosen_idx = -1;
        for (size_t j = 0; j < Q_r; j++)
        {
            double *transposed_Q = transpose(Q, Q_r, Q_c);
            double *q_point = transposed_Q + j*Q_c; //begin of line q_point of size P_x
            double dist = element_wise_reduce(p_point, q_point, 1, P_c, 1, Q_c,
                            squared_norm_2, add, add); //norm 2 between 2 vectors
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

double default_kernel(double a)
{
    if (a == 1) //fucking W_error
        return 1;
    else
        return 1;
} 

std::tuple<double*, std::vector<double>> compute_cross_variance(double *P, double *Q,
    std::vector<std::tuple<size_t, int>> correspondences, size_t P_r, size_t P_c,
    size_t Q_r, size_t Q_c, double (*kernel)(double a)) //set default function to lambda function??
{
    if (kernel == nullptr)
        kernel = &default_kernel;
    double *cov = (double*)calloc(4, sizeof(double));
    std::vector<double> exclude_indices = {};
    for (auto tup : correspondences)
    {
        auto i = std::get<0>(tup);
        auto j = std::get<1>(tup);
        double *transposed_P = transpose(P, P_r, P_c); 
        double *transposed_Q = transpose(Q, Q_r, Q_c);
        double *q_point = transposed_Q + j*Q_c;
        double *p_point = transposed_P + i*P_c;
        double weight = kernel(*p_point - *q_point);
        std::cout << weight << std::endl;
        if (weight < 0.01)
            exclude_indices.push_back(i);
        double **doted_points = nullptr;
        dot_product(doted_points, q_point, transpose(P, P_r, P_c), Q_r, Q_c, P_c, P_r); //dim of Q_r * P_r
        free(doted_points);
        double **weighted_points = nullptr; //multiply by the weight 
        element_wise_op(weighted_points, &weight, *doted_points, 1, 1, Q_r, P_r, mult);
        free(weighted_points);
        element_wise_op(&cov, cov, *weighted_points, 2, 2, Q_r, P_r, add);
    }
    return std::make_tuple(cov, exclude_indices);
}
