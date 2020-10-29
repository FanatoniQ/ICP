#include <stdio.h>
#include <iostream>
#include <iomanip>

// CPU
#include "libCSV/csv.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/alg.hpp"
#include "libalg/print.hpp"
#include "error.hpp"

// GPU
#include "libgpualg/euclidist.cuh"
#include "error.cuh"

void test_euclidist(double *d_Pt, double *d_Qt, size_t pitch, size_t width, size_t height)
{
    int threads = 4;
    double dist = sqrt(cuda_squared_norm_2(d_Pt, d_Qt, width, height, pitch, threads));
    std::cerr << "GPU squared mean diff: " << dist << std::endl;
}

int main(int argc, char **argv)
{
    runtime_assert(argc == 3, "Usage: ./testgpueuclid file1, file2");

    std::string h{};
    size_t Plines, Pcols, Qlines, Qcols;
    double *h_P = readCSV(argv[1], h, Plines, Pcols);
    double *h_Pt = transpose(h_P, Plines, Pcols);
    double *h_Q = readCSV(argv[2], h, Qlines, Qcols);
    double *h_Qt = transpose(h_Q, Qlines, Qcols);

    print_matrix(std::cerr, h_P, Pcols, Plines);
    print_matrix(std::cerr, h_Q, Qcols, Qlines);

    auto P = CPUMatrix(h_Pt, Pcols, Plines);
    auto Q = CPUMatrix(h_Qt, Qcols, Qlines);

    auto cpuEuclid = P.euclidianDistance(Q);

    // device memory
    double *d_Pt;
    size_t pitch;
    size_t width = Plines, height = Pcols;
    cudaMallocPitch(&d_Pt, &pitch, width * sizeof(double), height * sizeof(double));
    cudaCheckError();
    cudaMemcpy2D(d_Pt, pitch, h_Pt, width * sizeof(double), width * sizeof(double), height, cudaMemcpyHostToDevice);
    cudaCheckError();

    double *d_Qt;
    width = Qlines, height = Qcols;
    cudaMallocPitch(&d_Qt, &pitch, width * sizeof(double), height * sizeof(double));
    cudaCheckError();
    cudaMemcpy2D(d_Qt, pitch, h_Qt, width * sizeof(double), width * sizeof(double), height, cudaMemcpyHostToDevice);
    cudaCheckError();

    test_euclidist(d_Pt, d_Qt, pitch, width, height);

    std::cerr << "CPU squared mean diff: " << cpuEuclid << std::endl;
    std::cerr << "SUCCESS" << std::endl;
    cudaFree(d_Pt);
    cudaCheckError();
}
