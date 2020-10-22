#include <stdio.h>
#include <iostream>
#include <iomanip>

#include "libCSV/csv.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/alg.hpp"
#include "libalg/print.hpp"
#include "error.hpp"

#define cudaCheckError() {                                                                       \
        cudaError_t e=cudaGetLastError();                                                        \
        if(e!=cudaSuccess) {                                                                     \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }

__global__ void print_kernel()
{
    printf("Hello from block %d, thread %d\n", blockIdx.y * 10 + blockIdx.x, threadIdx.x);
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
 ** \param width the number of values in a line
 ** \param height the number of lines
 **/
__global__ void dumb_sum_kernel(const char *d_A, double *d_sumA, int pitch, int width, int height)
{
    int idx = threadIdx.x;
    if (idx >= height)
       return;
    double *line = (double *)(d_A + idx * pitch);
    d_sumA[idx] = get_line_sum(line, width);
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
 ** \param width the number of values in a line
 ** \param height the number of lines
 **/
__global__ void dumb_mean_kernel(const char *d_A, double *d_meanA, int pitch, int width, int height)
{
    int idx = threadIdx.x;
    if (idx >= height)
        return;
    double *line = (double *)(d_A + idx * pitch);
    d_meanA[idx] = get_line_mean(line, width);
}

__global__ void tree_reduce_sum_kernel(const double *d_A, double *d_sumA, int pitch, int width, int height, int reducepitch)
{
    extern __shared__ double s_data[]; // s_data is of size blockDim.x
    int threadid = threadIdx.x; // thread id in the block
    int lineid = blockIdx.y; // line
    int dataid = blockIdx.x * blockDim.x + threadIdx.x; // column
    if (dataid >= width || lineid >= height) {
	s_data[threadid] = 0; // prevent other threads from adding uninit values
        return;
    }
    double *d_Aline = (double *)((char *)d_A + lineid * pitch);
    // each thread copies to shared memory
    s_data[threadid] = d_Aline[dataid];
    __syncthreads();
    // each thread will reduce with one other shared data element in the middle right part of s_data
    for (size_t stride = blockDim.x / 2; stride > 0; stride = stride >> 1)
    {
        if (threadid < stride) // a lot of threads are idle...
             s_data[threadid] += s_data[threadid + stride];
	__syncthreads();
    }
    //printf("Reduce Index: %d\n", blockIdx.x + reducepitch * lineid);
    double *d_sumAline = (double *)((char *)d_sumA + lineid * reducepitch);
    if (threadid == 0)
        d_sumAline[blockIdx.x] = s_data[0];
    //d_sumA[blockIdx.x + reducepitch * lineid] = s_data[0]; // we store at pos x,y the partial mean
    //d_sumA[blockIdx.x + (height / blockDim.x) * lineid] = s_data[0]; // we store at pos x,y the partial mean
}

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

    //print_kernel<<<2, 3>>>();

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

    std::cerr << "GPU Mean : " << std::endl;
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
