#include <assert.h>

#include "error.hpp"

#include "libgpualg/euclidist.cuh"
#include "libgpualg/mean.cuh"
#include "error.cuh"

__global__ void euclidist_kernel(double *d_A, double *d_B, double *d_res, int pitch, int width, int height, int reducepitch)
{
    int threadid = threadIdx.x; // thread id in the block
    int r = blockIdx.y; // rows
    int c = blockDim.x * blockIdx.x + threadIdx.x; // cols
    extern __shared__ float s_data[];

    if (r >= width || c >= height)
    {
        s_data[threadid] = 0;
        return;
    }

    s_data[threadid] = (d_A[r + width * c] - d_B[r + width * c]) * (d_A[r + width * c] - d_B[r + width * c]);
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        assert(is_power_of_2(stride));
        if(threadid < stride)
            s_data[threadid] += s_data[threadid + stride];
        __syncthreads();
    }

    if(threadid == 0)
        d_res[blockIdx.x] = s_data[0];
}
