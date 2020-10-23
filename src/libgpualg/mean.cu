#include "error.cuh"
#include "libgpualg/mean.cuh"


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