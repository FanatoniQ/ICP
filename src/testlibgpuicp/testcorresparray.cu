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

int main(int argc, char **argv)
{
    std::string f1Header{};
    size_t Qlines, Qcols, Plines, Pcols;
    //size_t Plines, Pcols;
    //___readCSV(f, f1Header);
    double* Pt = readCSV(argv[1], f1Header, Plines, Pcols);
    for (int i = 0; i < 30; i++)
        std::cout << Pt[i] << std::endl;
    double* Qt = readCSV(argv[2], f1Header, Qlines, Qcols);

    double* d_P, * d_Q;

    cudaMalloc(&d_P, sizeof(double) * Plines * Pcols);
    cudaMalloc(&d_Q, sizeof(double) * Qlines * Qcols);

    cudaMemcpy(d_P, Pt, sizeof(double) * Pcols * Plines, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, Qt, sizeof(double) * Qcols * Qlines, cudaMemcpyHostToDevice);

    unsigned int* d_array_correspondances;
    cudaMalloc(&d_array_correspondances, sizeof(unsigned int) * Plines);

    get_array_correspondences(d_array_correspondances, d_P, d_Q, Plines, Pcols, Qlines, Qcols);

    cudaMemcpy(Qt, d_array_correspondances, sizeof(unsigned int) * Qcols, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 30; i++)
        std::cout << Qt[i] << std::endl;

    cudaFree(d_P);
    cudaFree(d_Q);
    cudaFree(d_array_correspondances);
    free(Pt);
    free(Qt);
}
