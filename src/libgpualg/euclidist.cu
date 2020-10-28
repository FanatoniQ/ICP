#include <assert.h>

#include "error.hpp"

#include "libgpualg/euclidist.cuh"
#include "libgpualg/mean.cuh"
#include "error.cuh"

__global__ void euclidist_kernel(double *d_A, double *d_B, double *d_res, int pitch, int width, int height, int reducepitch)
{
    int threadid = threadIdx.x; // thread id in the block
    int lineid = blockIdx.y; // rows
    int dataid = blockIdx.x * blockDim.x + threadIdx.x; // cols
    extern __shared__ double s_data[];

    if (dataid >= width || lineid >= height)
    {
        s_data[threadid] = 0;
        return;
    }

    double* d_Aline = (double*)((char*)d_A + lineid * pitch);
    double* d_Bline = (double*)((char*)d_B + lineid * pitch);
    double tmp = d_Aline[dataid] - d_Bline[dataid];
    s_data[threadid] = tmp * tmp;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        assert(is_power_of_2(stride));
        if(threadid < stride)
            s_data[threadid] += s_data[threadid + stride];
        __syncthreads();
    }

    double* d_resline = (double*)((char*)d_res + lineid * reducepitch);
    if(threadid == 0)
        d_resline[blockIdx.x] = s_data[0];
}
