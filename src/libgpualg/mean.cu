#include <iostream>
#include <stdio.h>
#include <assert.h>

#include "error.cuh"
#include "libgpualg/mean.cuh"

unsigned int get_next_power_of_2(unsigned int value)
{
    if (is_power_of_2(value))
        return value;
    if (value < 2)
        return 2;
    if (value < 4)
        return 4;
    if (value < 8)
        return 8;
    if (value < 16)
        return 16;
    if (value < 32)
        return 32;
    if (value < 64)
        return 64;
    if (value < 128)
        return 128;
    if (value < 256)
        return 256;
    if (value < 512)
        return 512;
    if (value < 1024)
	return 1024;
    fprintf(stderr, "Warning: get_next_power_of_two returns a number greater than 1024");
    unsigned int r = 1024;
    while (r < value)
        r <<= 1;
    return r;
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

// computes column sum (pitch is NOT in bytes)
__device__ __host__ double get_column_sum(const double *A, int colidx, int nblines, int pitch)
{
    int i;
    double sum = 0;
    for (i = 0; i < nblines; ++i)
        sum += A[i * pitch + colidx];
    return sum;
}

// computes column mean (pitch is NOT in bytes)
__device__ __host__ double get_column_mean(const double *A, int colidx, int nblines, int pitch)
{
    return get_column_sum(A, colidx, nblines, pitch) / nblines;
}

/** DUMB reduction **/

/** AXIS = 1 : summing each columns together for all lines **/

__global__ void dumb_sum_kernel(const char *d_A, double *d_sumA, int pitch, int width, int height)
{
    int idx = threadIdx.x;
    if (idx >= height)
       return;
    double *line = (double *)(d_A + idx * pitch);
    d_sumA[idx] = get_line_sum(line, width);
}

__global__ void dumb_mean_kernel(const char *d_A, double *d_meanA, int pitch, int width, int height)
{
    int idx = threadIdx.x;
    if (idx >= height)
        return;
    double *line = (double *)(d_A + idx * pitch);
    d_meanA[idx] = get_line_mean(line, width);
}

/** AXIS = 0 summing each lines together for  all columns **/

__global__ void dumb_sum_kernel_0(const double *d_A, double *d_sumA, int pitch, int width, int height)
{
    int idx = threadIdx.x;
    if (idx >= width)
        return;
    //printf("%d - %d\n", blockIdx.x, idx);
    d_sumA[idx] = get_column_sum(d_A, idx, height, pitch);
}

__global__ void dumb_mean_kernel_0(const double *d_A, double *d_sumA, int pitch, int width, int height)
{
    int idx = threadIdx.x;
    if (idx >= width)
        return;
    //printf("%d - %d | %d - %d\n",  blockIdx.y, blockIdx.x, idx, threadIdx.y);
    //assert(idx >= height);
    d_sumA[idx] = get_column_mean(d_A, idx, height, pitch);
}

/** TREE reduction **/

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
    // each thread copies to shared memory
    double *d_Aline = (double *)((char *)d_A + lineid * pitch);
    s_data[threadid] = d_Aline[dataid];
    __syncthreads();
    // each thread will reduce with one other shared data element in the middle right part of s_data
    for (size_t stride = blockDim.x / 2; stride > 0; stride = stride >> 1)
    {
        assert(is_power_of_2(stride)); // if not power of 2 ...
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

__global__ void tree_reduce_mean_kernel(const double *d_A, double *d_sumA, int pitch, int width, int height, int reducepitch)
{
    extern __shared__ double s_data[]; // s_data is of size blockDim.x
    int threadid = threadIdx.x; // thread id in the block
    int lineid = blockIdx.y; // line
    int dataid = blockIdx.x * blockDim.x + threadIdx.x; // column
    if (dataid >= width || lineid >= height) {
        s_data[threadid] = 0; // prevent other threads from adding uninit values
        return;
    }
    // each thread copies to shared memory
    double *d_Aline = (double *)((char *)d_A + lineid * pitch);
    s_data[threadid] = d_Aline[dataid];
    __syncthreads();
    // each thread will reduce with one other shared data element in the middle right part of s_data
    for (size_t stride = blockDim.x / 2; stride > 0; stride = stride >> 1)
    {
        assert(is_power_of_2(stride)); // if not power of 2 ...
        if (threadid < stride) // a lot of threads are idle...
             s_data[threadid] += s_data[threadid + stride];
        __syncthreads();
    }
    double *d_sumAline = (double *)((char *)d_sumA + lineid * reducepitch);
    if (threadid == 0)
        d_sumAline[blockIdx.x] = s_data[0] / width;
}

__global__ void tree_reduce_sum_kernel_0(const double *d_A, double *d_sumA, int pitch, int width, int height, int reducepitch)
{
    extern __shared__ double s_data[]; // s_data is of size blockDim.y
    int threadid = threadIdx.x; // thread id in the block
    int lineid = blockIdx.y * blockDim.x + threadIdx.x; // line
    int dataid = blockIdx.x; // column
    if (dataid >= width || lineid >= height) {
        s_data[threadid] = 0; // prevent other threads from adding uninit values
        return;
    }
    // each thread copies to shared memory
    double *d_Aline = (double *)((char *)d_A + lineid * pitch);
    s_data[threadid] = d_Aline[dataid];
    //printf("line: %d ,column: %d ,value: %lf\n", lineid, dataid, s_data[threadid]);
    __syncthreads();
    // each thread will reduce with one other shared data element in the middle right part of s_data
    for (size_t stride = blockDim.x / 2; stride > 0; stride = stride >> 1)
    {
        assert(is_power_of_2(stride)); // if not power of 2 ...
        if (threadid < stride) // a lot of threads are idle...
             s_data[threadid] += s_data[threadid + stride];
        __syncthreads();
    }
    //printf("Reduce Index: %d\n", blockIdx.x + reducepitch * lineid);
    double *d_sumAline = (double *)((char *)d_sumA + blockIdx.y * reducepitch);
    if (threadid == 0)
        d_sumAline[blockIdx.x] = s_data[0];
    //d_sumA[blockIdx.x + reducepitch * lineid] = s_data[0]; // we store at pos x,y the partial mean
    //d_sumA[blockIdx.x + (height / blockDim.x) * lineid] = s_data[0]; // we store at pos x,y the partial mean
}


__global__ void tree_reduce_mean_kernel_0(const double *d_A, double *d_sumA, int pitch, int width, int height, int reducepitch, size_t denom)
{
    extern __shared__ double s_data[]; // s_data is of size blockDim.y
    int threadid = threadIdx.x; // thread id in the block
    int lineid = blockIdx.y * blockDim.x + threadIdx.x; // line
    int dataid = blockIdx.x; // column
    if (dataid >= width || lineid >= height) {
        s_data[threadid] = 0; // prevent other threads from adding uninit values
        return;
    }
    // each thread copies to shared memory
    double *d_Aline = (double *)((char *)d_A + lineid * pitch);
    s_data[threadid] = d_Aline[dataid];
    __syncthreads();
    // each thread will reduce with one other shared data element in the middle right part of s_data
    for (size_t stride = blockDim.x / 2; stride > 0; stride = stride >> 1)
    {
        assert(is_power_of_2(stride)); // if not power of 2 ...
        if (threadid < stride) // a lot of threads are idle...
             s_data[threadid] += s_data[threadid + stride];
        __syncthreads();
    }
    double *d_sumAline = (double *)((char *)d_sumA + blockIdx.y * reducepitch);
    if (threadid == 0)
        d_sumAline[blockIdx.x] = s_data[0] / denom;
}

__host__ void reduce_0(enum MatrixReduceOP op, double *d_A, double **d_sum, size_t width, size_t height, size_t pitch, size_t *reducepitch, int threads)
{
    runtime_assert(op == MatrixReduceOP::SUM || op == MatrixReduceOP::MEAN, "Invalid reduce operation !");
    threads = get_next_power_of_2(threads);
    int nbblocksPerColumn = std::ceil((float)height / threads); // each block column treats partial one column sum
    dim3 blocks(width, nbblocksPerColumn); // we have width columns of nbblocksPerColumn

    if (*d_sum == nullptr)
    {
        // ALLOCATING DEVICE MEMORY
        // TODO: This is too much memory we should use cudaMalloc when we have a high number of lines
        cudaMallocPitch(d_sum, reducepitch, width * sizeof(double), nbblocksPerColumn);
        cudaCheckError();
        cudaMemset2D(*d_sum, *reducepitch, 0, width * sizeof(double), nbblocksPerColumn);
        cudaCheckError();
    }

    // LAUNCHING KERNEL
    std::cerr << "reducepitch: " << *reducepitch << " pitch: " << pitch << std::endl;
    std::cerr << "nbthreads: " << threads << " nbcolumns: " << blocks.x << " nbblocksPerColumns: " << blocks.y << std::endl;
    if (nbblocksPerColumn > 1)
    {
        tree_reduce_sum_kernel_0<<<blocks, threads, threads * sizeof(double)>>>(d_A, *d_sum, pitch, width, height, *reducepitch);
    } else {
        if (op == MatrixReduceOP::MEAN)
            tree_reduce_mean_kernel_0<<<blocks, threads, threads * sizeof(double)>>>(d_A, *d_sum, pitch, width, height, *reducepitch, height);
        else if (op == MatrixReduceOP::SUM)
            tree_reduce_sum_kernel_0<<<blocks, threads, threads * sizeof(double)>>>(d_A, *d_sum, pitch, width, height, *reducepitch);
    }
    cudaDeviceSynchronize();
    cudaCheckError();

    // We call the kernel a second time instead if multiple blocks per column
    // second call to reduce d_sum, nbthreads is nbblockPerColumn, height is nbblocksPerColumn, width is nbcols
    // Watch out ! do not use threads, blocks, width or pitch afterwards to reference d_A
    if (nbblocksPerColumn > 1)
    {
        pitch = *reducepitch;
        threads = nbblocksPerColumn;
        threads = get_next_power_of_2(threads);
        blocks = dim3(width, 1); // 1,height
        //height = nbblocksPerColumn;
        std::cerr << "reducepitch: " << *reducepitch << " pitch: " << pitch << std::endl;
        std::cerr << "nbthreads: " << threads << " nbcolumns: " << blocks.x << " nbblocksPerColumns: " << blocks.y << std::endl;
        if (op == MatrixReduceOP::MEAN)
            tree_reduce_mean_kernel_0<<<blocks, threads, threads * sizeof(double)>>>(*d_sum, *d_sum, pitch, width, nbblocksPerColumn, *reducepitch, height);
        else if (op == MatrixReduceOP::SUM)
            tree_reduce_sum_kernel_0<<<blocks, threads, threads * sizeof(double)>>>(*d_sum, *d_sum, pitch, width, nbblocksPerColumn, *reducepitch);
        cudaDeviceSynchronize();
        cudaCheckError();
    }
}

__host__ void mean_0(double *d_A, double **d_sum, size_t width, size_t height, size_t pitch, size_t *reducepitch, int threads)
{
    reduce_0(MatrixReduceOP::MEAN, d_A, d_sum, width, height, pitch, reducepitch, threads);
}