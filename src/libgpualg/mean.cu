#include "error.cuh"
#include "libgpualg/mean.cuh"

#include <assert.h>

// TODO: REMOVE this, or not it is just in assert afterall
#define is_power_of_2(x) (x & (x-1)) == 0

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
