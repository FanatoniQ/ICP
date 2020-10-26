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

// \deprecated use CPUMatrix::euclidianDistance instead
/**
double three_dim_norm(CPUMatrix A)
{
    //if (A.getDim1() != 3)
    //    throw std::invalid_argument("Matrix not of dim 3");
    double r = 0;
    for (size_t i = 0; i < A.getDim1(); ++i)
        r += pow2(A(0, i));
    if (A.getDim1() == 3)
        runtime_assert((std::pow(A(0, 0), 2) + std::pow(A(0, 1), 2) + std::pow(A(0, 2), 2)) == r, "FATAL");
    auto norm = A.squared_norm(-1);
    runtime_assert(norm.getDim0() == 1 && norm.getDim1() == 1, "INVALID NORM SIZE ! FATAL ERROR");
    double res = norm(0, 0);
    runtime_assert(r == res, "INVALID NORM ! FATAL ERROR");
    //return std::pow(A(0, 0), 2) + std::pow(A(0, 1), 2) + std::pow(A(0, 2), 2);
    return r;
}**/

std::vector<std::tuple<size_t, int>> get_correspondence_indices(double *P, double *Q,
                                                                size_t P_r, size_t P_c, size_t Q_r, size_t Q_c)
{
    std::vector<std::tuple<size_t, int>> correspondances = {};
    for (size_t i = 0; i < P_r; i++)
    {
        //double *transposed_P = transpose(P, P_r, P_c);
        //double *p_point = transposed_P + i * P_c; //begin of line p_point of size P_x
        double *p_point = P + i * P_c;
        double min_dist = std::numeric_limits<double>::max();
        int chosen_idx = -1;
        for (size_t j = 0; j < Q_r; j++)
        {
            //double *transposed_Q = transpose(Q, Q_r, Q_c);
            //double *q_point = transposed_Q + j * Q_c; //begin of line q_point of size P_x
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

std::vector<std::tuple<size_t, int>> get_correspondence_indices(CPUMatrix &P, CPUMatrix &Q)
{
    std::vector<std::tuple<size_t, int>> correspondances = {};
    for (size_t i = 0; i < P.getDim0(); i++)
    {
        auto p_point = P.getLine(i);
        double min_dist = std::numeric_limits<double>::max();
        int chosen_idx = -1;
        for (size_t j = 0; j < Q.getDim0(); j++)
        {
            auto q_point = Q.getLine(j);
            double dist = std::sqrt(p_point.euclidianDistance(q_point));
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

double default_kernel(CPUMatrix a)
{
    UNUSED(a);
    return 1;
}

double default_kernel(double a)
{
    UNUSED(a);
    return 1;
}

std::tuple<CPUMatrix, std::vector<double>> compute_cross_variance(CPUMatrix &P, CPUMatrix &Q,
                                                                  const std::vector<std::tuple<size_t, int>> &correspondences, double (*kernel)(CPUMatrix a))
{
    if (kernel == nullptr)
        kernel = &default_kernel;
    CPUMatrix cov = CPUMatrix(P.getDim1(), P.getDim1());
    std::vector<double> exclude_indices = {};
    for (auto tup : correspondences)
    {
        auto i = std::get<0>(tup);
        auto j = std::get<1>(tup);
        CPUView q_point = Q.getLine(j);
        CPUView p_point = P.getLine(i);
        double weight = kernel(p_point - q_point);

        if (weight < 0.01)
            exclude_indices.push_back(i);

        CPUMatrix doted_points = q_point.transpose().dot(p_point);
        doted_points *= weight;
        cov += doted_points;
    }
    return std::make_tuple(std::move(cov), exclude_indices);
}

std::tuple<CPUMatrix, std::vector<double>, std::vector<std::tuple<size_t, int>>> icp(CPUMatrix &P, CPUMatrix &Q, unsigned iterations)
{
    // Center data P and Q
    auto Q_center = Q.mean(0);
    Q -= Q_center;

    std::vector<std::tuple<size_t, int>> correps_values;
    std::vector<double> norm_values;
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


/**
// \deprecated
std::tuple<double *, std::vector<double>> compute_cross_variance(double *P, double *Q,
                                                                 std::vector<std::tuple<size_t, int>> correspondences, size_t P_r, size_t P_c,
                                                                 size_t Q_r, size_t Q_c, double (*kernel)(double a)) //set default function to lambda function??
{
    if (kernel == nullptr)
        kernel = &default_kernel;
    double *cov = (double *)calloc(4, sizeof(double));
    std::vector<double> exclude_indices = {};
    for (auto tup : correspondences)
    {
        auto i = std::get<0>(tup);
        auto j = std::get<1>(tup);
        double *transposed_P = transpose(P, P_r, P_c);
        double *transposed_Q = transpose(Q, Q_r, Q_c);
        double *q_point = transposed_Q + j * Q_c;
        double *p_point = transposed_P + i * P_c;
        double weight = kernel(*p_point - *q_point);

        if (weight < 0.01)
            exclude_indices.push_back(i);
        double **doted_points = nullptr;
        dot_product(doted_points, q_point, transpose(P, P_r, P_c), Q_r, Q_c, P_c, P_r); //dim of Q_r * P_r
        free(doted_points);
        double **weighted_points = nullptr; //multiply by the weight
        size_t Wdim0, Wdim1;                // use high level API instead
        element_wise_op(weighted_points, &weight, *doted_points, 1, 1, Q_r, P_r, Wdim0, Wdim1, mult);
        free(weighted_points);
        size_t Wcov0, Wcov1; // should be 2, use high level API instead
        element_wise_op(&cov, cov, *weighted_points, 2, 2, Q_r, P_r, Wcov0, Wcov1, add);
    }
    return std::make_tuple(cov, exclude_indices);
}**/

