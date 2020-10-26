#include <assert.h>

#include "error.hpp"

#include "libgpualg/euclidist.cuh"
#include "libgpualg/mean.cuh"
#include "error.cuh"

__global__ void euclidist_kernel(double *d_A, double *d_B, double *d_res, int pitch, int width, int height, int reducepitch)
{
    int r = blockDim.x * blockIdx.x + threadIdx.x; // rows
    int c = blockDim.y * blockIdx.y + threadIdx.y; // cols 
    extern __shared__ float s_data[];

    if (r >= width || c >= height)
    {
        s_data[threadIdx.x] = 0;
        return;
    }

    s_data[threadIdx.x] = (d_A[r + width * c] - d_B[r + width * c]) * (d_A[r + width * c] - d_B[r + width * c]);
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        assert(is_power_of_2(stride));
        if(threadIdx.x < stride)
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        __syncthreads();
    }

    if(threadIdx.x == 0)
        d_res[r] = s_data[0];
}
