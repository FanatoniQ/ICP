#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <iostream>
#include <limits>
#include <float.h>
#include <assert.h>

// CPU
#include "libCSV/csv.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/CPUView.hpp"
#include "libalg/alg.hpp"
#include "libalg/print.hpp"
#include "error.hpp"


// GPU
#include "libgpualg/mean.cuh"
#include "error.cuh"
#include "libgpuicp/corresp.cuh"
#include "libgpuicp/dist.cuh"
#include "libgpuicp/corresp.cuh"
#include "libgpuicp/crosscov.cuh"


int main(int argc, char **argv)
{
    std::string f1Header{};
    size_t Qlines, Qcols, Plines, Pcols;

    double* Pt = readCSV(argv[1], f1Header, Plines, Pcols);
    double* Qt = readCSV(argv[2], f1Header, Qlines, Qcols);

    double* d_P, * d_Q, *d_cov;

    cudaMalloc(&d_P, sizeof(double) * Plines * Pcols);
    cudaMalloc(&d_Q, sizeof(double) * Qlines * Qcols);

    cudaMemcpy(d_P, Pt, sizeof(double) * Pcols * Plines, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, Qt, sizeof(double) * Qcols * Qlines, cudaMemcpyHostToDevice);

    unsigned int* d_array_correspondances;
    cudaMalloc(&d_array_correspondances, sizeof(unsigned int) * Plines);

    get_array_correspondences(d_array_correspondances, d_P, d_Q, Plines, Pcols, Qlines, Qcols);


    cudaMalloc(&d_cov, sizeof(double) * Qcols * Pcols);
    get_array_cross_cov(d_cov, d_array_correspondances, d_P, d_Q, Plines, Pcols, Qlines, Qcols);
    double* h_cov = (double*)malloc(Pcols * Qcols * sizeof(double));
    cudaMemcpy(h_cov, d_cov, Pcols * Qcols * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < Pcols * Qcols; i++)
    {
        std::cout << h_cov[i] << " ";
        if (i % 3)
            std::cout << std::endl;
    }

    cudaFree(d_P);
    cudaFree(d_Q);
    cudaFree(d_array_correspondances);
    free(h_cov);
    free(Pt);
    free(Qt);
}
