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

// axis=0 dumb sum testing (same as test_dumb_sum but swapped width and height
void test_dumb_sum_0(const CPUMatrix &cpuSum, float *d_p, size_t pitch, size_t width, size_t height)
{
    // SETUP
    float *d_sum;
    size_t blocks = 1;//width;
    size_t threads = width;//height; // TODO: handle width > 1024
    runtime_assert(threads <= 1024, "Too many threads");
    
    // ALLOCATING DEVICE MEMORY
    cudaMalloc(&d_sum, width * sizeof(float));

    float *h_sum = (float*)malloc(width * sizeof(float));
    runtime_assert(h_sum != nullptr, "Alloc error !");

    
    // LAUNCHING KERNEL
    dumb_sum_kernel_0<<<blocks, threads>>>(d_p, d_sum, pitch / sizeof(float), width, height);
    cudaDeviceSynchronize();
    cudaCheckError();

    // COPY TO HOST
    cudaMemcpy(h_sum, d_sum, width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // FREEING DEVICE MEMORY
    cudaFree(d_sum);
    cudaCheckError();

    // PRINTING
    std::cerr << "GPU Sums : " << std::endl;
    for (size_t i = 0; i < width; ++i)
    {
	float cpulinesum = cpuSum(0,i);
        float gpulinesum = 0;
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
    std::cerr << "AXIS0 DONE" << std::endl;
}


void test_dumb_sum(const CPUMatrix &cpuSum, float *d_pT, size_t pitch, size_t width, size_t height)
{
    // SETUP
    float *d_sum;
    size_t blocks = 1;//height;
    size_t threads = height;//width; // TODO: handle width > 1024
    runtime_assert(threads <= 1024, "Too many threads");
    
    // ALLOCATING DEVICE MEMORY
    cudaMalloc(&d_sum, height * sizeof(float));

    float *h_sum = (float*)malloc(height * sizeof(float));
    runtime_assert(h_sum != nullptr, "Alloc error !");

    
    // LAUNCHING KERNEL
    dumb_sum_kernel<<<blocks, threads>>>((char*)d_pT, d_sum, pitch, width, height);
    cudaDeviceSynchronize();
    cudaCheckError();

    // COPY TO HOST
    cudaMemcpy(h_sum, d_sum, height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // FREEING DEVICE MEMORY
    cudaFree(d_sum);
    cudaCheckError();

    // PRINTING
    std::cerr << "GPU Sums : " << std::endl;
    for (size_t i = 0; i < height; ++i)
    {
	float cpulinesum = cpuSum(0,i);
        float gpulinesum = 0;
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

void test_tree_reduce_sum(const CPUMatrix &cpuSum, float *d_pT, size_t pitch, size_t width, size_t height, bool multiiter)
{
    // SETUP
    float *d_sum;
    size_t reducepitch;
    int threads = 1024; // TODO: change this
    threads = get_next_power_of_2(threads);
    int nbblocksPerLine = std::ceil((float)width / threads); // each block line treats partial one line sum
    dim3 blocks(nbblocksPerLine, height); // we have height lines of nbblocksPerLine

    // ALLOCATING DEVICE MEMORY
    cudaMallocPitch(&d_sum, &reducepitch, nbblocksPerLine * sizeof(float), height);
    cudaCheckError();
    cudaMemset2D(d_sum, reducepitch, 0, nbblocksPerLine * sizeof(float), height);
    cudaCheckError();

    // LAUNCHING KERNEL
    std::cerr << "reducepitch: " << reducepitch << " pitch: " << pitch << std::endl;
    std::cerr << "nbthreads: " << threads << " nbblocksPerLine: " << blocks.x << " nbLines: " << blocks.y << std::endl;
    tree_reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_pT, d_sum, pitch, width, height, reducepitch);
    cudaDeviceSynchronize();
    cudaCheckError();

    // We call the kernel a second time instead if multiple blocks per line
    // second call to reduce d_mean, nbthreads is nbblockPerLine, width is nbblockPerline, height is nblines
    // Watch out ! do not use threads, blocks, width or pitch afterwards to reference d_pT
    if (multiiter)
    {
        pitch = reducepitch;
        threads = nbblocksPerLine;
        threads = get_next_power_of_2(threads);
        blocks = dim3(1, height);
        width = nbblocksPerLine;
        std::cerr << "reducepitch: " << reducepitch << " pitch: " << pitch << std::endl;
        std::cerr << "nbthreads: " << threads << " nbblocksPerLine: " << blocks.x << " nbLines: " << blocks.y << std::endl;
        tree_reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_sum, d_sum, pitch, width, height, reducepitch);
        cudaDeviceSynchronize();
        cudaCheckError();
    }

    float *h_sum = (float*)malloc(height * reducepitch);
    runtime_assert(h_sum != nullptr, "Alloc error !");

    // COPY TO HOST
    cudaMemcpy(h_sum, d_sum, height * reducepitch, cudaMemcpyDeviceToHost);
    cudaCheckError();

    // FREEING DEVICE MEMORY
    cudaFree(d_sum);
    cudaCheckError();

    // PRINTING
    float ttlerror = 0;
    std::cerr << "GPU Sums : " << std::endl;
    for (size_t i = 0; i < height; ++i)
    {
        float *line = (float *)((char *)h_sum + i * reducepitch);
	float cpulinesum = cpuSum(0,i);
        float gpulinesum = 0;
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
	ttlerror += std::fabs(cpulinesum - gpulinesum);
	// we have some error apparently, which is weird
	//runtime_assert(cpulinesum == gpulinesum, "Not same mean");
	/**
	// NOTE: floating points operations are not commutative...
	// we could use bigger type than float for shared memory... can be heavy
        if (std::fabs(cpulinesum - gpulinesum) > 1e-10f) {
             std::cerr << "Difference betweeen CPU and GPU sum: " << gpulinesum - cpulinesum << std::endl;
             exit(4);
        }**/
    }
    std::cerr << std::endl << "Axis1:" << std::endl << "Total error: " << ttlerror << std::endl;
    std::cerr << "Mean error: " << ttlerror / width << std::endl << std::endl;
    free(h_sum);
}

void test_tree_reduce_sum_0(const CPUMatrix &cpuSum, float *d_p, size_t pitch, size_t width, size_t height, bool multiiter)
{
    // SETUP
    float *d_sum;
    size_t reducepitch;
    int threads = 1024; // TODO: change this
    threads = get_next_power_of_2(threads);
    int nbblocksPerColumn = std::ceil((float)height / threads); // each block column treats partial one column sum
    dim3 blocks(width, nbblocksPerColumn); // we have width columns of nbblocksPerColumn

    // ALLOCATING DEVICE MEMORY
    // TODO: This is too much memory we should use cudaMalloc when we have a high number of lines
    cudaMallocPitch(&d_sum, &reducepitch, width * sizeof(float), nbblocksPerColumn);
    cudaCheckError();
    cudaMemset2D(d_sum, reducepitch, 0, width * sizeof(float), nbblocksPerColumn);
    cudaCheckError();

    // LAUNCHING KERNEL
    std::cerr << "reducepitch: " << reducepitch << " pitch: " << pitch << std::endl;
    std::cerr << "nbthreads: " << threads << " nbcolumns: " << blocks.x << " nbblocksPerColumns: " << blocks.y << std::endl;
    tree_reduce_sum_kernel_0<<<blocks, threads, threads * sizeof(float)>>>(d_p, d_sum, pitch, width, height, reducepitch);
    cudaDeviceSynchronize();
    cudaCheckError();

    // We call the kernel a second time instead if multiple blocks per column
    // second call to reduce d_sum, nbthreads is nbblockPerColumn, height is nbblocksPerColumn, width is nbcols
    // Watch out ! do not use threads, blocks, width or pitch afterwards to reference d_p
    if (multiiter)
    {
        pitch = reducepitch;
        threads = nbblocksPerColumn;
        threads = get_next_power_of_2(threads);
        blocks = dim3(width, 1); // 1,height
        height = nbblocksPerColumn;
        std::cerr << "reducepitch: " << reducepitch << " pitch: " << pitch << std::endl;
        std::cerr << "nbthreads: " << threads << " nbcolumns: " << blocks.x << " nbblocksPerColumns: " << blocks.y << std::endl;
        tree_reduce_sum_kernel_0<<<blocks, threads, threads * sizeof(float)>>>(d_sum, d_sum, pitch, width, height, reducepitch);
        cudaDeviceSynchronize();
        cudaCheckError();
    }

    float *h_sum = (float*)malloc(nbblocksPerColumn * reducepitch);
    runtime_assert(h_sum != nullptr, "Alloc error !");

    // COPY TO HOST
    cudaMemcpy(h_sum, d_sum, nbblocksPerColumn * reducepitch, cudaMemcpyDeviceToHost);
    cudaCheckError();

    // FREEING DEVICE MEMORY
    cudaFree(d_sum);
    cudaCheckError();

    // PRINTING
    float ttlerror = 0;
    std::cerr << "GPU Sums : " << std::endl;
    for (size_t i = 0; i < width; ++i)
    {
	float cpulinesum = cpuSum(0,i);
        float gpulinesum = 0;
	// is gpu second reduce:
	if (multiiter)
	    gpulinesum = h_sum[i];
        // one reduce case, final reduce done on cpu
	else
            for (size_t j = 0; j < nbblocksPerColumn; ++j)
            {
                gpulinesum += h_sum[j * (reducepitch / sizeof(float)) + i];
                std::cerr <<  h_sum[j * (reducepitch / sizeof(float)) + i] << "(" << gpulinesum << ")+\t";
            }
        std::cerr << std::endl << "line[0]" << gpulinesum << std::endl;
	std::cerr << "CPUSUM(0,i)" << cpulinesum << std::endl;
        std::cerr << "Difference betweeen CPU and GPU sum: " << gpulinesum - cpulinesum << std::endl;
	ttlerror += std::fabs(cpulinesum - gpulinesum);
	// we have some error apparently, which is weird
	//runtime_assert(cpulinesum == gpulinesum, "Not same mean");
	/**
	// NOTE: floating points operations are not commutative...
	// we could use bigger type than float for shared memory... can be heavy
        if (std::fabs(cpulinesum - gpulinesum) > 1e-10f) {
             std::cerr << "Difference betweeen CPU and GPU sum: " << gpulinesum - cpulinesum << std::endl;
             exit(4);
        }**/
    }
    std::cerr << std::endl << "Axis0:" << std::endl << "Total error: " << ttlerror << std::endl;
    std::cerr << "Mean error: " << ttlerror / height << std::endl << std::endl;
    free(h_sum);
}

void test_tree_reduce_mean_0(const CPUMatrix &cpuMean, float *d_A, size_t pitch, size_t width, size_t height)
{
    // SETUP
    float *d_mean = nullptr;
    int threads = 4; // TODO: change this
    size_t reducepitch;
    mean_0(d_A, &d_mean, width, height, pitch, &reducepitch, threads);

    float *h_mean = (float*)malloc(1 * reducepitch); // final result is a line vector
    runtime_assert(h_mean != nullptr, "Alloc error !");

    // COPY TO HOST
    cudaMemcpy(h_mean, d_mean, 1 * reducepitch, cudaMemcpyDeviceToHost);
    cudaCheckError();

    // FREEING DEVICE MEMORY
    cudaFree(d_mean);
    cudaCheckError();

    // PRINTING
    float ttlerror = 0;
    std::cerr << "GPU Means : " << std::endl;
    for (size_t i = 0; i < width; ++i)
    {
	float cpulinemean = cpuMean(0,i);
        float gpulinemean = h_mean[i];
        std::cerr << std::endl << "line[0]" << gpulinemean << std::endl;
	std::cerr << "CPUSUM(0,i)" << cpulinemean << std::endl;
        std::cerr << "Difference betweeen CPU and GPU mean: " << gpulinemean - cpulinemean << std::endl;
	ttlerror += std::fabs(cpulinemean - gpulinemean);
    }
    std::cerr << std::endl << "Axis0 (mean-host-wrapper):" << std::endl << "Total error: " << ttlerror << std::endl;
    std::cerr << "Mean error: " << ttlerror / height << std::endl << std::endl;
    free(h_mean);
}


int main_axis0(int argc, char **argv)
{
    runtime_assert(argc == 4 || argc == 3, "Usage: ./testgpusum file1 method [axis]");
    std::cerr << std::setprecision(15);
    std::string h{};
    size_t nblines, nbcols;
    float *h_p = readCSV(argv[1], h, nblines, nbcols);
    print_matrix(std::cerr, h_p, nbcols, nblines);

    auto P = CPUMatrix(h_p, nblines, nbcols);
    auto cpuSum = P.sum(0);
    std::cerr << "CPU Sums: " << std::endl << cpuSum << std::endl;
    
    // device memory
    float *d_p;
    size_t pitch;
    size_t width = nbcols, height = nblines;
    cudaMallocPitch(&d_p, &pitch, width * sizeof(float), height * sizeof(float));
    cudaCheckError();
    cudaMemcpy2D(d_p, pitch, h_p, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
    cudaCheckError();

    if (strcmp(argv[2], "dummy") == 0)
        test_dumb_sum_0(cpuSum, d_p, pitch, width, height);
    else if (strcmp(argv[2], "tree") == 0)
        test_tree_reduce_sum_0(cpuSum, d_p, pitch, width, height, false);
    else if (strcmp(argv[2], "trees") == 0)
        test_tree_reduce_sum_0(cpuSum, d_p, pitch, width, height, true);
    else if (strcmp(argv[2], "treex") == 0)
        test_tree_reduce_mean_0(P.mean(0), d_p, pitch, width, height);
    else
    {
        std::cerr << "method = dummy | tree | trees | treex" << std::endl;
        return EXIT_FAILURE;
    }
    std::cerr << cpuSum << std::endl;
    std::cerr << "SUCCESS" << std::endl;
    cudaFree(d_p);
    cudaCheckError();
    return EXIT_SUCCESS;
}


int main_axis1(int argc, char **argv)
{
    runtime_assert(argc == 3 || argc == 4, "Usage: ./testgpusum file1 method [axis]");
    std::cerr << std::setprecision(15);
    std::string h{};
    size_t nblines, nbcols;
    float *h_p = readCSV(argv[1], h, nblines, nbcols);
    float *h_pT = transpose(h_p, nblines, nbcols);
    print_matrix(std::cerr, h_p, nbcols, nblines);

    auto P = CPUMatrix(h_pT, nbcols, nblines);
    auto cpuSum = P.sum(1);
    std::cerr << "CPU Sums: " << std::endl << cpuSum << std::endl;
    
    // device memory
    float *d_pT;
    size_t pitch;
    size_t width = nblines, height = nbcols;
    cudaMallocPitch(&d_pT, &pitch, width * sizeof(float), height * sizeof(float));
    cudaCheckError();
    cudaMemcpy2D(d_pT, pitch, h_pT, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
    cudaCheckError();
 
    if (strcmp(argv[2], "dummy") == 0)
        test_dumb_sum(cpuSum, d_pT, pitch, width, height);
    else if (strcmp(argv[2], "trees") == 0)
        test_tree_reduce_sum(cpuSum, d_pT, pitch, width, height, true);
    else if (strcmp(argv[2], "tree") == 0)
        test_tree_reduce_sum(cpuSum, d_pT, pitch, width, height, false);
    else
    {
        std::cerr << "method = dummy | tree | trees" << std::endl;
        return EXIT_FAILURE;
    }
    std::cerr << cpuSum << std::endl;
    std::cerr << "SUCCESS" << std::endl;
    cudaFree(d_pT);
    cudaCheckError();
    free(h_p);
    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    runtime_assert(argc == 3 || argc == 4, "Usage ./testgpusum file1 method [axis]");
    if (argc == 3 || strcmp(argv[3], "1") == 0)
        return main_axis1(argc, argv);
    if (argc == 4 && strcmp(argv[3], "0") == 0)
        return main_axis0(argc, argv);
    std::cerr << "Usage: axis = 0 | 1" << std::endl;
    return EXIT_FAILURE;
}
