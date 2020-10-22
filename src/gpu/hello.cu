#include <stdio.h>
#include <iostream>

#include "libCSV/csv.hpp"

__global__ void print_kernel()
{
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

// TODO: REMOVE ME since useless
__global__ void print_matrix_kernel(char *d_A, int pitch, int nbvals)
{
    int j;
    int idx = threadIdx.x;
    double *line = (double*)(d_A + idx * pitch);
    printf("Line %d:\n", idx);
    for (j = 0; j < nbvals; ++j) {
        //printf("%6.2f\t", (double)(d_A[idx * pitch + j * sizeof(double)]));
        printf("%6.2f\t", line[j]);
	__syncthreads();
    }
    printf("\n");
}

// computes line sum
__device__ __host__ double get_line_sum(const double *line, int nbvals)
{
    int j;
    double sum = 0;
    for (j = 0; j < nbvals; ++j)
        sum += line[j];
    return sum;
}

// computes line mean
__device__ __host__ double get_line_mean(const double *line, int nbvals)
{
    return get_line_sum(line, nbvals) / nbvals;
}

/**
 ** \brief dumb_sum_kernel a dummy kernel to compute axis=1 sum
 ** each thread will sum one line with coallesced access then write
 ** the sum to global device memory at address d_sumA[idx] with
 ** idx being the threadIdx.x. In order to get the sum for each lines
 ** we need to launch nblines kernels. Each thread will call the
 ** get_line_sum function.
 ** \note kernel can be improved by launching multiple threads per line
 ** and potentially divide and conquer strategy
 **
 ** \param d_A the device pointer
 ** \param d_sumA the device sum pointer used to store sum results
 ** \param pitch the pitch in bytes
 ** \param nbvals the number of values in a line
 **/
__global__ void dumb_sum_kernel(char *d_A, double *d_sumA, int pitch, int nbvals)
{
    //int j;
    int idx = threadIdx.x;
    //double sum = 0;
    double *line = (double *)(d_A + idx * pitch);
    //for (j = 0; j < nbvals; ++j)
    //    sum += line[j];
    //d_sumA[idx] = sum;
    d_sumA[idx] = get_line_sum(line, nbvals);
}

/**
 ** \brief dumb_mean_kernel a dummy kernel to compute axis=1 mean
 ** This kernel is basically the same as the dumb_sum_kernel, but
 ** each thread calls the get_line_mean function instead of get_line_sum.
 ** In order to get the mean for each lines we need to launch nblines kernels.
 **
 ** \param d_A the device pointer
 ** \param d_meanA the device mean pointer used to store mean results
 ** \param pitch the pitch in bytes
 ** \param nbvals the number of values in a line
 **/
__global__ void dumb_mean_kernel(char *d_A, double *d_meanA, int pitch, int nbvals)
{
    int idx = threadIdx.x;
    double *line = (double *)(d_A + idx * pitch);
    d_meanA[idx] = get_line_mean(line, nbvals); //get_line_sum(line, nbvals) / nbvals;
}

// TODO: use libalg
__host__ double *transpose(double *h_A, size_t dim0, size_t dim1)
{
    size_t i,j;
    double *h_AT;
    h_AT = (double *)malloc(dim0 * dim1 * sizeof(double));
    if (h_AT == nullptr)
    {
        std::cerr << "Alloc failed !" << std::endl;
        exit(1);
    }
    for (i = 0; i < dim0; ++i)
    {
        for (j = 0; j < dim1; ++j)
        {
             h_AT[j * dim0 + i] = h_A[i * dim1 + j];
        }
    }
    return h_AT;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: ./hello file1" << std::endl;
        exit(1);
    }
    std::string h{};
    size_t nblines, nbcols;
    double *h_p = readCSV(argv[1], h, nblines, nbcols);
    double *h_pT = transpose(h_p, nblines, nbcols);
    for (size_t i = 0; i < nbcols; ++i)
    {
        for (size_t j = 0; j < nblines; ++j)
        {
             std::cerr << h_pT[i * nblines + j] << "\t";
        }
        std::cerr << std::endl << std::endl;
    }
    //print_kernel<<<2, 3>>>();

    // device memory
    double *d_pT;
    size_t pitch;
    size_t width = nblines, height = nbcols;
    cudaMallocPitch(&d_pT, &pitch, width * sizeof(double), height * sizeof(double)); // TODO: error checking
    cudaMemcpy2D(d_pT, pitch, h_pT, width * sizeof(double), width * sizeof(double), height, cudaMemcpyHostToDevice); // TODO: error checking
    double *d_mean;
    cudaMalloc(&d_mean, height * sizeof(double)); // TODO: error checking

    // kernel launching
    //print_matrix_kernel<<<1, height>>>((char *)d_pT, pitch, width);
    //dumb_sum_kernel<<<1, height>>>((char*)d_pT, d_mean, pitch, width);
    dumb_mean_kernel<<<1, height>>>((char*)d_pT, d_mean, pitch, width);
    cudaDeviceSynchronize();

    // copy back to host memory
    double *h_mean;
    h_mean = (double*)malloc(height * sizeof(double));
    if (h_mean == nullptr)
        exit(1);
    cudaMemcpy(h_mean, d_mean, height * sizeof(double), cudaMemcpyDeviceToHost);

    std::cerr << "Mean : " << std::endl;
    for (size_t i = 0; i < height; ++i)
        std::cerr << h_mean[i] << std::endl;

    cudaFree(d_mean);
    cudaFree(d_pT);
    free(h_mean);
    free(h_p);
    free(h_pT);
    return EXIT_SUCCESS;
}
