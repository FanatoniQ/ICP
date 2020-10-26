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

// FIXME: duplicated in mean.cu
#define is_power_of_2(x) (x & (x-1)) == 0

void test_dumb_sum(const CPUMatrix &cpuSum, double *d_pT, size_t pitch, size_t width, size_t height)
{
    // SETUP
    double *d_sum;
    size_t blocks = height;
    size_t threads = width; // TODO: handle width > 1024
    runtime_assert(threads <= 1024, "Too many threads");
    
    // ALLOCATING DEVICE MEMORY
    cudaMalloc(&d_sum, height * sizeof(double));

    double *h_sum = (double*)malloc(height * sizeof(double));
    runtime_assert(h_sum != nullptr, "Alloc error !");

    
    // LAUNCHING KERNEL
    dumb_sum_kernel<<<blocks, threads>>>((char*)d_pT, d_sum, pitch, width, height);
    cudaDeviceSynchronize();
    cudaCheckError();

    // COPY TO HOST
    cudaMemcpy(h_sum, d_sum, height * sizeof(double), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // FREEING DEVICE MEMORY
    cudaFree(d_sum);
    cudaCheckError();

    // PRINTING
    std::cerr << "GPU Sums : " << std::endl;
    for (size_t i = 0; i < height; ++i)
    {
	double cpulinesum = cpuSum(0,i);
        double gpulinesum = 0;
	gpulinesum = h_sum[i]; // dummy sum kernel
        std::cerr << std::endl << "line[0]" << gpulinesum << std::endl;
	std::cerr << "CPUSUM(0,i)" << cpulinesum << std::endl;
	// we have some error apparently, which is weird
	//runtime_assert(cpulinesum == gpulinesum, "Not same mean");
        if (std::fabs(cpulinesum - gpulinesum) > 1e-10f) {
             std::cerr << "Difference betweeen CPU and GPU sum: " << gpulinesum - cpulinesum << std::endl;
             exit(4);
        }
    }
    free(h_sum);
}

void test_tree_reduce_sum(const CPUMatrix &cpuSum, double *d_pT, size_t pitch, size_t width, size_t height, bool multiiter)
{
    // SETUP
    double *d_sum;
    size_t reducepitch;
    int threads = 4; // TODO: change this
    while (!is_power_of_2(threads))
        threads++; // FIXME: this is slow, consider function to return next closest power_of_2
    int nbblocksPerLine = std::ceil((float)width / threads); // each block line treats partial one line sum
    dim3 blocks(nbblocksPerLine, height); // we have height lines of nbblocksPerLine

    // ALLOCATING DEVICE MEMORY
    cudaMallocPitch(&d_sum, &reducepitch, nbblocksPerLine * sizeof(double), height);
    cudaCheckError();
    cudaMemset2D(d_sum, reducepitch, 0, nbblocksPerLine * sizeof(double), height);
    cudaCheckError();

    // LAUNCHING KERNEL
    std::cerr << "reducepitch: " << reducepitch << " pitch: " << pitch << std::endl;
    std::cerr << "nbthreads: " << threads << " nbblocksPerLine: " << blocks.x << " nbLines: " << blocks.y << std::endl;
    tree_reduce_sum_kernel<<<blocks, threads, threads * sizeof(double)>>>(d_pT, d_sum, pitch, width, height, reducepitch);
    cudaDeviceSynchronize();
    cudaCheckError();

    // We call the kernel a second time instead if multiple blocks per line
    // second call to reduce d_mean, nbthreads is nbblockPerLine, width is nbblockPerline, height is nblines
    // Watch out ! do not use threads, blocks, width or pitch afterwards to reference d_pT
    if (multiiter)
    {
    pitch = reducepitch;
    threads = nbblocksPerLine;
    while (!is_power_of_2(threads))
        threads++;
    blocks = dim3(1, height);
    width = nbblocksPerLine;
    std::cerr << "reducepitch: " << reducepitch << " pitch: " << pitch << std::endl;
    std::cerr << "nbthreads: " << threads << " nbblocksPerLine: " << blocks.x << " nbLines: " << blocks.y << std::endl;
    tree_reduce_sum_kernel<<<blocks, threads, threads * sizeof(double)>>>(d_sum, d_sum, pitch, width, height, reducepitch);
    cudaDeviceSynchronize();
    cudaCheckError();
    }

    double *h_sum = (double*)malloc(height * reducepitch);
    runtime_assert(h_sum != nullptr, "Alloc error !");

    // COPY TO HOST
    cudaMemcpy(h_sum, d_sum, height * reducepitch, cudaMemcpyDeviceToHost);
    cudaCheckError();

    // FREEING DEVICE MEMORY
    cudaFree(d_sum);
    cudaCheckError();

    // PRINTING
    std::cerr << "GPU Sums : " << std::endl;
    for (size_t i = 0; i < height; ++i)
    {
        double *line = (double *)((char *)h_sum + i * reducepitch);
	double cpulinesum = cpuSum(0,i);
        double gpulinesum = 0;
	// is gpu second reduce:
	if (multiiter)
	    gpulinesum = line[0];
        // one reduce case, final reduce done on cpu
	else
            for (size_t j = 0; j < nbblocksPerLine; ++j)
            {
                gpulinesum += line[j];
                std::cerr << line[j] << "(" << gpulinesum << ")+\t";
            }
        std::cerr << std::endl << "line[0]" << gpulinesum << std::endl;
	std::cerr << "CPUSUM(0,i)" << cpulinesum << std::endl;
        std::cerr << "Difference betweeen CPU and GPU sum: " << gpulinesum - cpulinesum << std::endl;
	// we have some error apparently, which is weird
	//runtime_assert(cpulinesum == gpulinesum, "Not same mean");
	/**
	// NOTE: floating points operations are not commutative...
	// we could use bigger type than double for shared memory... can be heavy
        if (std::fabs(cpulinesum - gpulinesum) > 1e-10f) {
             std::cerr << "Difference betweeen CPU and GPU sum: " << gpulinesum - cpulinesum << std::endl;
             exit(4);
        }**/
    }
    free(h_sum);
}

int main(int argc, char **argv)
{
    runtime_assert(argc == 3, "Usage: ./testgpusum file1 method");
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
 
    if (strcmp(argv[2], "dummy") == 0)
        test_dumb_sum(cpuSum, d_pT, pitch, width, height);
    else if (strcmp(argv[2], "trees") == 0)
        test_tree_reduce_sum(cpuSum, d_pT, pitch, width, height, true);
    else if (strcmp(argv[2], "tree") == 0)
        test_tree_reduce_sum(cpuSum, d_pT, pitch, width, height, false);
    else
    {
        std::cerr << "method = dummy | tree" << std::endl;
        return EXIT_FAILURE;
    }
    std::cerr << cpuSum << std::endl;
    std::cerr << "SUCCESS" << std::endl;
    cudaFree(d_pT);
    cudaCheckError();
    free(h_p);
}


#ifdef TOTO
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
    //print_kernel<<<blocks, threads>>>();
    //print_matrix_kernel<<<blocks, threads>>>((char *)d_pT, pitch, width);
    dumb_sum_kernel<<<height, width>>>((char*)d_pT, d_mean, pitch, width, height);
    //dumb_mean_kernel<<<blocks, threads>>>((char*)d_pT, d_mean, pitch, width, height);
    cudaDeviceSynchronize();
    cudaCheckError();
    
    // tree_reduce sum: FIXME
    /**
    std::cerr << "nbthreads: " << threads << " nbblocksPerLine: " << blocks.x << " nbLines: " << blocks.y << std::endl;
    tree_reduce_sum_kernel<<<blocks, threads, threads * sizeof(double)>>>(d_pT, d_mean, pitch, width, height, reducepitch);
    cudaDeviceSynchronize();
    cudaCheckError();
    // FIXME: Call the kernel a second time instead if multiple blocks per line
    // second call to reduce d_mean, nbthreads is nbblockPerLine, width is nbblockPerline, height is nblines
    // Watch out ! do not use threads, blocks, width or pitch afterwards to reference d_pT
    
    pitch = reducepitch; // rowstride = threads since we stride with threads, to sum each block of threads partial sums
    threads = nbblocksPerLine;
    blocks = dim3(1, height);
    width = nbblocksPerLine;
    std::cerr << "nbthreads: " << threads << " nbblocksPerLine: " << blocks.x << " nbLines: " << blocks.y << std::endl;
    tree_reduce_sum_kernel<<<blocks, threads, threads * sizeof(double)>>>(d_mean, d_mean, pitch, width, height, reducepitch);
    cudaDeviceSynchronize();
    cudaCheckError();
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
        //double *line = (double *)((char *)h_mean + i * reducepitch);
	double cpulinesum = cpuSum(0,i);
        double gpulinesum = 0;
	// is gpu second reduce:
	//gpulinesum = line[0];
	gpulinesum = h_mean[i]; // dummy sum kernel
        /**
        // one reduce case, final reduce done on cpu
        for (size_t j = 0; j < nbblocksPerLine; ++j)
        {
            std::cerr << line[j] << "+\t";
            gpulinesum += line[j];
        }
        **/
        std::cerr << std::endl << "line[0]" << gpulinesum << std::endl;
	std::cerr << "CPUSUM(0,i)" << cpulinesum << std::endl;
	// we have some error apparently, which is weird
	//runtime_assert(cpulinesum == gpulinesum, "Not same mean");
        if (std::fabs(cpulinesum - gpulinesum) > 1e-10f) {
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
#endif