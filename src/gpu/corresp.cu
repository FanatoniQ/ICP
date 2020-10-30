#include <limits>
#include <float.h>
#include <assert.h>
#include <stdio.h>

#include "libgpualg/mean.cuh"
#include "gpu/corresp.cuh"
#include "error.cuh"

#define is_power_of_2(x) (x & (x-1)) == 0

__global__ void get_correspondences_kernel(ICPCorresp *d_dist,
    size_t dist_pitch, size_t dist_0, size_t dist_1)
{
    extern __shared__ ICPCorresp s_data[]; // s_data is of size blockDim.x
    unsigned int threadid = threadIdx.x; // thread id in the block
    unsigned int lineid = blockIdx.y; // line
    unsigned int dataid = blockIdx.x * blockDim.x + threadIdx.x; // column
    if (dataid >= dist_1 || lineid >= dist_0) {
        s_data[threadid] = { DBL_MAX,dataid };
        return;
    }
    // each thread copies to shared memory
    ICPCorresp *d_distline = (ICPCorresp *)((char *)d_dist + lineid * dist_pitch);
    s_data[threadid] = d_distline[dataid];
    __syncthreads();
    // each thread will reduce with one other shared data element in the middle right part of s_data
    for (size_t stride = blockDim.x / 2; stride > 0; stride = stride >> 1)
    {
        assert(is_power_of_2(stride)); // if not power of 2 ...
        if (threadid < stride) { // a lot of threads are idle...
            if (s_data[threadid + stride].dist < s_data[threadid].dist)
                s_data[threadid] = s_data[threadid + stride];
        }
        __syncthreads();
    }
    if (threadid == 0)
        d_distline[blockIdx.x] = s_data[0]; // or [0] since gridsize.x should be 1
}

__host__ void get_correspondences(ICPCorresp *d_dist,
    size_t dist_pitch, size_t dist_0, size_t dist_1, bool sync)
{
    dim3 gridsize(1, get_next_power_of_2(dist_0));
    dim3 blocksize(dist_1, 1);
    get_correspondences_kernel<<<gridsize, blocksize, blocksize.x * sizeof(ICPCorresp)>>>(d_dist, dist_pitch, dist_0, dist_1);
    if (sync) {
        cudaDeviceSynchronize();
        cudaCheckError();
    }
}
