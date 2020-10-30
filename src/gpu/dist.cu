#include <limits>
#include <float.h>
#include <assert.h>
#include <stdio.h>
#include <iostream>

#include "libgpualg/mean.cuh"
#include "gpu/corresp.cuh"
#include "error.cuh"
#include "error.hpp"

//gpu-icp-dist-matrix

__global__ void get_distances_kernel(const double *d_P, const double *d_Q, ICPCorresp *d_dist,
    size_t p_0, size_t p_1, size_t p_pitch,
    size_t q_0, size_t q_1, size_t q_pitch,
    size_t dist_0, size_t dist_1, size_t dist_pitch)
{
    //unsigned int threadid = threadIdx.x; // thread id in the block
    unsigned int colid = blockIdx.x * blockDim.x + threadIdx.x; // Q index
    unsigned int lineid = blockIdx.y * blockDim.y + threadIdx.y; // P index
    if (colid >= dist_1 || lineid >= dist_0) {
        return;
    }
    double *d_Pline = (double *)((char*)d_P + lineid * p_pitch);
    double *d_Qline = (double *)((char*)d_Q + colid * q_pitch);
    double dist = 0;
    double tmp;
    // TODO: for higher dimensions than 3, we can improve this by making each thread compute just one subtract^2
    // then reduce sum
    for (size_t i = 0; i < p_1; ++i)
    {
        tmp = d_Qline[i] - d_Pline[i];
        dist += tmp * tmp;
    }
    ICPCorresp *d_distLine = (ICPCorresp *)((char*)d_dist + lineid * dist_pitch);
    d_distLine[colid] = { dist, colid };
}

__host__ void get_distances(const double *d_P, const double *d_Q, ICPCorresp **d_dist,
    size_t p_0, size_t p_1, size_t p_pitch,
    size_t q_0, size_t q_1, size_t q_pitch,
    size_t dist_0, size_t dist_1, size_t *dist_pitch, bool sync)
{
    runtime_assert(p_1 == q_1 && dist_1 == q_0 && dist_0 == p_0, "Invalid shapes !");
    if (*d_dist == nullptr)
    {
        cudaMallocPitch(d_dist, dist_pitch, dist_1 * sizeof(ICPCorresp), dist_0);
        cudaCheckError();
    }
    
    dim3 blocksize(32, 32); // TODO: chanhe size
    dim3 gridsize(std::ceil((float)p_0 / blocksize.x), std::ceil((float)p_1 / blocksize.y));

    std::cerr << "gridsize.y: " << gridsize.y << " blocksize.y: " << blocksize.y << std::endl;
    std::cerr << "gridsize.x: " << gridsize.x << " blocksize.x: " << blocksize.x << std::endl;
    
    runtime_assert(gridsize.y * blocksize.y >= dist_0, "not enough to handle each P");
    runtime_assert(gridsize.x * blocksize.x >= dist_1, "not enough to handle each Q");
    
    get_distances_kernel<<<gridsize, blocksize>>>(d_P, d_Q, *d_dist, p_0, p_1, p_pitch, q_0, q_1, q_pitch, dist_0, dist_1, *dist_pitch);
    if (sync) {
        cudaDeviceSynchronize();
        cudaCheckError();
    }
}
