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
    // SETUP
    double *d_res;
    size_t reducepitch;
    int threads = 4;
    while (!is_power_of_2(threads))
        threads++;
    int nbblocksPerLine = std::ceil((float)width / threads);
    dim3 blocks(nbblocksPerLine, height);

    // ALLOCATING DEVICE MEMORY
    cudaMallocPitch(&d_res, &reducepitch, nbblocksPerLine * sizeof(double), height);
    cudaCheckError();
    cudaMemset2D(d_res, reducepitch, 0, nbblocksPerLine * sizeof(double), height);
    cudaCheckError();

    // LAUNCHING KERNEL
    std::cerr << "reducepitch: " << reducepitch << " pitch: " << pitch << std::endl;
    std::cerr << "nbthreads: " << threads << " nbblocksPerLine: " << blocks.x << " nbLines: " << blocks.y << std::endl;
    euclidist_kernel<<<blocks, threads, threads * sizeof(double)>>>(d_Pt, d_Qt, d_res, pitch, width, height, reducepitch);
    cudaDeviceSynchronize();
    cudaCheckError();

    double *h_res = (double*)malloc(height * reducepitch);
    runtime_assert(h_res != nullptr, "Alloc error !");

    // COPY TO HOST
    cudaMemcpy(h_res, d_res, height * reducepitch, cudaMemcpyDeviceToHost);
    cudaCheckError();

    // FREEING DEVICE MEMORY
    cudaFree(d_res);
    cudaCheckError();

    std::cerr << "GPU squared mean diff: " << h_res[0] << std::endl;
    free(h_res);
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

    auto P = CPUMatrix(h_P, Pcols, Plines);
    auto Q = CPUMatrix(h_Q, Qcols, Qlines);

    auto cpuEuclid = P.euclidianDistance(Q);

    // device memory
    double *d_Pt;
    size_t pitch;
    size_t width = Plines, height = Pcols;
    cudaCheckError();
    cudaMallocPitch(&d_Pt, &pitch, width * sizeof(double), height * sizeof(double));
    cudaMemcpy2D(d_Pt, pitch, h_Pt, width * sizeof(double), width * sizeof(double), height, cudaMemcpyHostToDevice);
    cudaCheckError();

    double *d_Qt;
    width = Qlines, height = Qcols;
    cudaCheckError();
    cudaMallocPitch(&d_Qt, &pitch, width * sizeof(double), height * sizeof(double));
    cudaMemcpy2D(d_Qt, pitch, h_Qt, width * sizeof(double), width * sizeof(double), height, cudaMemcpyHostToDevice);
    cudaCheckError();

    test_euclidist(d_Pt, d_Qt, pitch, width, height);

    std::cerr << "CPU squared mean diff: " << cpuEuclid << std::endl;
    std::cerr << "SUCCESS" << std::endl;
    cudaFree(d_Pt);
    cudaCheckError();
}
