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
    auto cpuSum = P.sum(1);
    std::cerr << "CPU Sums: " << std::endl << cpuSum << std::endl;

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
    tree_reduce_sum_kernel<<<blocks, threads, threads * sizeof(double)>>>(d_pT, d_mean, pitch, width, height, reducepitch);
    cudaDeviceSynchronize();
    cudaCheckError();
    // FIXME: Call the kernel a second time instead if multiple blocks per line
    // second call to reduce d_mean, nbthreads is nbblockPerLine, width is nbblockPerline, height is nblines
    // Watch out ! do not use threads, blocks, width or pitch afterwards to reference d_pT
    /**
    pitch = reducepitch; // rowstride = threads since we stride with threads, to sum each block of threads partial sums
    threads = nbblocksPerLine;
    blocks = dim3(1, height);
    width = nbblocksPerLine;
    std::cerr << "nbthreads: " << threads << " nbblocksPerLine: " << blocks.x << " nbLines: " << blocks.y << std::endl;
    tree_reduce_sum_kernel<<<blocks, threads, threads * sizeof(double)>>>(d_mean, d_mean, pitch, width, height, reducepitch);
    cudaDeviceSynchronize();
    cudaCheckError(); // FIXME: failure
    **/

    // copy back to host memory
    std::cerr << "Device -> Host" << std::endl;
    double *h_mean;
    h_mean = (double*)malloc(height * reducepitch);
    runtime_assert(h_mean != nullptr, "Alloc error !");
    cudaMemcpy(h_mean, d_mean, height * reducepitch, cudaMemcpyDeviceToHost);
    cudaCheckError();

    std::cerr << "GPU Sums : " << std::endl;
    for (size_t i = 0; i < height; ++i)
    {
        double *line = (double *)((char *)h_mean + i * reducepitch);
	double cpulinesum = cpuSum(0,i);
        double gpulinesum = 0;
	// is gpu second reduce:
	//gpulinesum = line[0];
        //**
        // one reduce case, final reduce done on cpu
        for (size_t j = 0; j < nbblocksPerLine; ++j)
        {
            std::cerr << line[j] << "+\t";
            gpulinesum += line[j];
        }
	//**/
        std::cerr << std::endl << "line[0]" << gpulinesum << std::endl;
	std::cerr << "CPUSUM(0,i)" << cpulinesum << std::endl;
	// we have some error apparently, which is weird
	//runtime_assert(cpulinesum == gpulinesum, "Not same mean");
        if (std::fabs(cpulinesum - gpulinesum) > 1e-13f) {
             std::cerr << "Difference betweeen CPU and GPU sum: " << gpulinesum - cpulinesum << std::endl;
             exit(4);
        }
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
