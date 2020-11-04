#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <iostream>

// CPU
#include "libCSV/csv.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/alg.hpp"
#include "libalg/print.hpp"
#include "error.hpp"

// GPU
#include "libgpualg/mean.cuh"
#include "error.cuh"
#include "gpu/icp.cuh"
#include "gpu/gputimer.cuh"

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

// Implementation with CPUMAtrix
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

int main(int argc, char **argv)
{
    runtime_assert(argc >= 4, "Usage: ./GPUICP file1 file2 nbiters");
    std::string f1Header{};
    size_t Qlines, Qcols, Plines, Pcols;

    double *Pt = readCSV(argv[1], f1Header, Plines, Pcols);
    CPUMatrix P = CPUMatrix(Pt, Plines, Pcols);

    double *Qt = readCSV(argv[2], f1Header, Qlines, Qcols);
    CPUMatrix Q = CPUMatrix(Qt, Qlines, Qcols);

    unsigned int nbiters = std::stoi(argv[3]);
    CPUMatrix P_res;

    if (argc == 5 && strcmp(argv[4], "-batch") == 0)
         P_res = icp_gpu(P, Q, nbiters);
    else
         P_res = icp_gpu_optimized(P, Q, nbiters);

    std::cout << "Squared actual mean diff: " << Q.euclidianDistance(P_res) << std::endl;
    P_res.write_in_file("point_cloud_result");
}
