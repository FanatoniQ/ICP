#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <iostream>
#include <limits>
#include <float.h>

// CPU
#include "libCSV/csv.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/alg.hpp"
#include "libalg/print.hpp"
#include "error.hpp"


// GPU
#include "libgpualg/mean.cuh"
#include "error.cuh"
//#include "gpu/icp.cuh"
#include "libgpuicp/corresp.cuh"
#include "libgpuicp/crosscov.cuh"

int main(int argc, char **argv)
{
    std::string f1Header{};
    size_t Qlines, Qcols, Plines, Pcols;
    double* Pt = readCSV(argv[1], f1Header, Plines, Pcols);
    double* Qt = readCSV(argv[2], f1Header, Qlines, Qcols);
    double* d_P, * d_Q;

    unsigned int p_pitch = Pcols * sizeof(double);
    cudaMalloc(&d_P, sizeof(double) * Plines * Pcols);
    unsigned int q_pitch = Qcols * sizeof(double);
    cudaMalloc(&d_Q, sizeof(double) * Qlines * Qcols);

    cudaMemcpy(d_P, Pt, sizeof(double) * Pcols * Plines, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, Qt, sizeof(double) * Qcols * Qlines, cudaMemcpyHostToDevice);

    unsigned int* d_array_correspondances;
    cudaMalloc(&d_array_correspondances, sizeof(unsigned int) * Plines);

    get_array_correspondences(d_array_correspondances, d_P, d_Q, Plines, Pcols, Qlines, Qcols);

    unsigned int r_0 = Plines, r_1 = Pcols * Qcols;
    double* d_R = nullptr;
    size_t r_pitch;

    get_array_cross_covs_flattened(d_P, d_Q, &d_R, d_array_correspondances,
        Plines, Pcols, p_pitch,
        Qlines, Qcols, q_pitch,
        r_0, r_1, &r_pitch,
        Pcols, true);

    cudaFree(d_R);
    cudaFree(d_P);
    cudaFree(d_Q);
    cudaFree(d_array_correspondances);
    free(Pt);
    free(Qt);
}
