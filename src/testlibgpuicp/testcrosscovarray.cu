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
//#include "gpu/icp.cuh"
#include "libgpuicp/corresp.cuh"
#include "libgpuicp/dist.cuh"
#include "libgpuicp/corresp.cuh"
#include "libgpuicp/crosscov.cuh"


int main(int argc, char **argv)
{
    std::string f1Header{};
    size_t Qlines, Qcols, Plines, Pcols;
    //size_t Plines, Pcols;
    //___readCSV(f, f1Header);
    float* Pt = readCSV(argv[1], f1Header, Plines, Pcols);
    float* Qt = readCSV(argv[2], f1Header, Qlines, Qcols);

    float* d_P, * d_Q, *d_cov;

    cudaMalloc(&d_P, sizeof(float) * Plines * Pcols);
    cudaMalloc(&d_Q, sizeof(float) * Qlines * Qcols);

    cudaMemcpy(d_P, Pt, sizeof(float) * Pcols * Plines, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, Qt, sizeof(float) * Qcols * Qlines, cudaMemcpyHostToDevice);

    unsigned int* d_array_correspondances;
    cudaMalloc(&d_array_correspondances, sizeof(unsigned int) * Plines);

    get_array_correspondences(d_array_correspondances, d_P, d_Q, Plines, Pcols, Qlines, Qcols);

    /*
    unsigned int* h_array_correspondances = (unsigned int*)malloc(Plines * sizeof(unsigned int));
    cudaMemcpy(h_array_correspondances, d_array_correspondances, Plines * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < Plines; i++)
        std::cout << i << " ->" << h_array_correspondances[i] << std::endl;
    */

    cudaMalloc(&d_cov, sizeof(float) * Qcols * Pcols);
    get_array_cross_cov(d_cov, d_array_correspondances, d_P, d_Q, Plines, Pcols, Qlines, Qcols);
    float* h_cov = (float*)malloc(Pcols * Qcols * sizeof(float));
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
