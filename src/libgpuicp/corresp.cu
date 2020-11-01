#include <limits>
#include <float.h>
#include <assert.h>
#include <stdio.h>
#include <iostream>

#include "libgpualg/mean.cuh"
#include "libgpuicp/corresp.cuh"
#include "error.cuh"

#include "error.hpp"

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
    //printf("%u,%u\n", lineid, dataid);
    // each thread copies to shared memory
    ICPCorresp *d_distline = (ICPCorresp *)((char *)d_dist + lineid * dist_pitch);
    s_data[threadid] = d_distline[dataid];
    //printf("%u: %lf,%d \n", lineid, d_distline[dataid].dist, d_distline[dataid].id);
    //s_data[threadid].dist = d_distline[dataid].dist;
    //s_data[threadid].id = d_distline[dataid].id;
    __syncthreads();
    // each thread will reduce with one other shared data element in the middle right part of s_data
    for (size_t stride = blockDim.x / 2; stride > 0; stride = stride >> 1)
    {
        assert(is_power_of_2(stride)); // if not power of 2 ...
        if (threadid < stride) { // a lot of threads are idle...
            if (s_data[threadid + stride].dist < s_data[threadid].dist) {
                s_data[threadid] = s_data[threadid + stride];
                //printf("line:%u| %lu: %lf,%d < %u: %lf,%d \n", lineid, threadid + stride, s_data[threadid + stride].dist, s_data[threadid + stride].id, threadid, s_data[threadid].dist, s_data[threadid].id);
                //s_data[threadid].dist = s_data[threadid + stride].dist;
                //s_data[threadid].id = s_data[threadid + stride].id;
            }
        }
        __syncthreads();
    }
    if (threadid == 0) {
        d_distline[blockIdx.x] = s_data[0]; // or [0] since gridsize.x should be 1
        //printf("FINAL: %u: %lf,%d \n", lineid, d_distline[blockIdx.x].dist, d_distline[blockIdx.x].id);
    }
}

/**
// deprecated
__host__ void get_correspondences(ICPCorresp *d_dist,
    size_t dist_pitch, size_t dist_0, size_t dist_1, bool sync)
{
    //dim3 gridsize(1, dist_0);
    //dim3 blocksize(get_next_power_of_2(dist_1), 1);
    //size_t nb = get_next_power_of_2(dist_1);
    dim3 blocksize(1024, 1);
    dim3 gridsize(std::ceil((float)dist_1 / blocksize.x), dist_0);
    std::cerr << std::endl << "gridsize.y: " << gridsize.y << std::endl;
    std::cerr << "blocksize.x: " << blocksize.x << std::endl;
    get_correspondences_kernel<<<gridsize, blocksize, blocksize.x * sizeof(ICPCorresp)>>>(d_dist, dist_pitch, dist_0, dist_1);
    if (sync) {
        cudaDeviceSynchronize();
        cudaCheckError();
    }
}**/

__host__ void get_correspondences(ICPCorresp *d_dist,
    size_t dist_pitch, size_t dist_0, size_t dist_1, bool sync, size_t threads)
{
    threads = get_next_power_of_2(threads);
    int nbblocksPerLine = std::ceil((float)dist_1 / threads); // each block column treats partial one line sum
    dim3 blocks(nbblocksPerLine, dist_0); // we have dist_0 lines of nbblocksPerLine

    // LAUNCHING KERNEL
    std::cerr << "nbthreads: " << threads << " nblines: " << blocks.y << " nbblocksPerLine: " << blocks.x << std::endl;
    get_correspondences_kernel<<<blocks, threads, threads * sizeof(ICPCorresp)>>>(d_dist, dist_pitch, dist_0, dist_1);
    cudaDeviceSynchronize();
    cudaCheckError();

    // We call the kernel a second time instead if multiple blocks per line
    // second call to reduce d_dist, nbthreads is nbblockPerLine, dist_0 is nbblocksPerLine
    if (nbblocksPerLine > 1)
    {
        threads = nbblocksPerLine;
        threads = get_next_power_of_2(threads);
        dist_1 = nbblocksPerLine;
        blocks = dim3(1, dist_0);
        
        std::cerr << "nbthreads: " << threads << " nblines: " << blocks.y << " nbblocksPerLine: " << blocks.x << std::endl;
        get_correspondences_kernel<<<blocks, threads, threads * sizeof(ICPCorresp)>>>(d_dist, dist_pitch, dist_0, dist_1);
        cudaDeviceSynchronize();
        cudaCheckError();
    }
}

//d_corresp declarer en global
__device__ size_t *d_array_correspondances;

__global__ void get_array_correspondences_kernel(double *P, double *Q, size_t P_row, size_t P_col, size_t Q_row, size_t Q_col)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (!(P_col == Q_col && P_row == Q_row && P_col == 3)) //device code does not support exception handling
        return;
    if (index >= P_row)
        return;

    double *p_point = P + index * P_col;
    double min_dist = DBL_MAX;
    size_t chosen_idx = 0;

    for (size_t y = 0; y < P_row; y++) //make sure this is nb rows
    {
        double *q_point = Q + y * Q_col;
        double dist = std::sqrt(std::pow(p_point[0] - q_point[0], 2) + std::pow(p_point[1] - q_point[1], 2) + std::pow(p_point[2] - q_point[2], 2));
        
        if (dist < min_dist)
        {
            min_dist = dist;
            chosen_idx = y;
        }
        d_array_correspondances[index] = chosen_idx;
    }
}

__host__ void get_array_correspondences(double *P, double *Q,
    size_t P_row, size_t P_col, size_t Q_row, size_t Q_col)
{
    dim3 blocksize(1024, 1);
    dim3 gridsize(std::ceil((float)P_row / blocksize.x), 1);
    std::cerr << std::endl << "gridsize.x: " << gridsize.x << std::endl;
    std::cerr << "blocksize.x: " << blocksize.x << std::endl;

    cudaMalloc(&d_array_correspondances, sizeof(unsigned short int) * P_row);
    get_array_correspondences_kernel<<<gridsize, blocksize>>>(P, Q, P_row, P_col, Q_row, Q_col);
    cudaDeviceSynchronize();
    cudaCheckError();
}
