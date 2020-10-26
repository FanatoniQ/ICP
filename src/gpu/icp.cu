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

/*
__global__ void get_correspondence_indices(CPUMatrix &P, CPUMatrix &Q, std::vector<std::tuple<size_t, int>> &correspondances)
{
    int i = blockIdx.x*width+threadIdx.x; //P getDim0
    int j = blockIdx.x*width+threadIdx.x; //Q getDim0

    if(i< P.getDim0() && j < Q.getDim0()) {
        auto p_point = P.getLine(i);
        double min_dist = std::numeric_limits<double>::max();
        int chosen_idx = -1;

        auto q_point = Q.getLine(j);
        double dist = std::sqrt(p_point.euclidianDistance(q_point));
        if (dist < min_dist)
        {
            min_dist = dist;
            chosen_idx = j;
        }
        correspondances.push_back(std::make_tuple(i, chosen_idx));
    }
    //
    for (size_t i = 0; i < P.getDim0(); i++)
    {
        auto p_point = P.getLine(i);
        double min_dist = std::numeric_limits<double>::max();
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
    
}
*/

__global__ void compute_cross_variance(CPUMatrix &P, CPUMatrix &Q, const std::vector<std::tuple<size_t, int>> &correspondences,
                            double (*kernel)(CPUMatrix a), std::tuple<CPUMatrix, std::vector<double>> &res) //pas besoin de cuda en fait c'est des op de base
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
