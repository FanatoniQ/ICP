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
#include "libgpualg/mean.cuh"
#include "error.cuh"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: ./hello file1" << std::endl;
        exit(1);
    }
    std::cerr << std::setprecision(15);
    std::string h{};
    size_t nblines, nbcols;
    double *h_p = readCSV(argv[1], h, nblines, nbcols);
    double *h_pT = transpose(h_p, nblines, nbcols);
    print_matrix(std::cerr, h_p, nbcols, nblines);

    auto P = CPUMatrix(h_pT, nbcols, nblines);
    std::cerr << "CPU Sums: " << std::endl << P.sum(1) << P.sum(0) << std::endl;

    // device memory
    double *d_pT;
    size_t pitch;
    size_t width = nblines, height = nbcols;
    cudaMallocPitch(&d_pT, &pitch, width * sizeof(double), height * sizeof(double));
    cudaCheckError();
    cudaMemcpy2D(d_pT, pitch, h_pT, width * sizeof(double), width * sizeof(double), height, cudaMemcpyHostToDevice);
    cudaCheckError();
    double *d_mean;
    //cudaMalloc(&d_mean, height * sizeof(double));
    
    // kernel launching
    //int blocks = 2;
    //int threads = std::ceil((float)height / blocks); // max is 1024 threads per blocks !
    //runtime_assert(blocks * threads >= height, "Invalid Grid Shape !");
    int threads = 4; // TODO: change this
    int nbblocksPerLine = std::ceil((float)width / threads); // each block line treats one line sum
    dim3 blocks(nbblocksPerLine, height); // we have lines line of nbblocksPerLine
    size_t reducepitch;
    cudaMallocPitch(&d_mean, &reducepitch, nbblocksPerLine * sizeof(double), height);
    cudaCheckError();
    cudaMemset2D(d_mean, reducepitch, 0, nbblocksPerLine * sizeof(double), height);
    cudaCheckError();
    std::cerr << "nbthreads: " << threads << " nbblocksPerLine: " << blocks.x << " nbLines: " << blocks.y << std::endl;
    //print_kernel<<<blocks, threads>>>();
    //print_matrix_kernel<<<blocks, threads>>>((char *)d_pT, pitch, width);
    //dumb_sum_kernel<<<blocks, threads>>>((char*)d_pT, d_mean, pitch, width);
    //dumb_mean_kernel<<<blocks, threads>>>((char*)d_pT, d_mean, pitch, width, height);
    tree_reduce_sum_kernel<<<blocks, threads, threads>>>(d_pT, d_mean, pitch, width, height, reducepitch); // FIXME: illegal mem access
    cudaDeviceSynchronize();
    cudaCheckError();

    // copy back to host memory
    double *h_mean;
    h_mean = (double*)malloc(height * reducepitch);
    runtime_assert(h_mean != nullptr, "Alloc error !");
    cudaMemcpy(h_mean, d_mean, height * reducepitch, cudaMemcpyDeviceToHost);
    cudaCheckError();

    std::cerr << "GPU Sums : " << std::endl;
    // FIXME: Call the kernel a second time instead if multiple blocks per line
    for (size_t i = 0; i < height; ++i)
    {
        double v = 0;
        double *line = (double *)((char *)h_mean + i * reducepitch);
        for (size_t j = 0; j < nbblocksPerLine; ++j)
        {
            std::cerr << "[]: " << line[j] << "\t";
            v += line[j];
        }
        std::cerr << v << std::endl;
        //std::cerr << h_mean[i] << "\t";
    }

    cudaFree(d_mean);
    cudaCheckError();
    cudaFree(d_pT);
    cudaCheckError();
    //free(h_mean);
    free(h_p);
    //free(h_pT);
    return EXIT_SUCCESS;
}