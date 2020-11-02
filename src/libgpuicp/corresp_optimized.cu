#include <limits>
#include <float.h>
#include <assert.h>
#include <stdio.h>
#include <iostream>

#include "libgpualg/mean.cuh"
#include "libgpuicp/corresp.cuh"
#include "libgpuicp/corresp_optimized.cuh"
#include "error.cuh"

#include "error.hpp"

/**
 ** \brief optimized version of get_array_correspondences_kernel, we have one block per point in p, 1024 threads blocks
 ** which loops if more than 1024 points in q. ICPCorresp structure is used for storing dist,idq, this kernel min reduces
 ** in place this array and repeats the process
 ** we launch with gridsize.x == p_0, gridsize.y == 1; blocksize.x == 1024, blocksize.y == 1
 ** <<<dim3(p_0,1), dim3(1024,1), sizeof(ICPCorresp) * 1024 + sizeof(ICPCorresp) + 3 * sizeof(double)>>>
 ** \note C.C. >=3.0 => maximum 2^31 - 1 blocks on x dimension and 65535 on y and z
 ** \note blockIdx.x is used as P index
 ** \note threadIdx.x is not related to blockIdx.x, it is a threadid used for Q index
 ** \note we loop in case blockDim.x is not enough
 ** \note we do multiple iterations with nbiters since we want to share the data between threads
 **/
__global__ void get_array_correspondences_optimized_kernel(unsigned int *d_array_correspondances,
    const double *d_P, const double *d_Q, unsigned int P_row, unsigned int P_col, unsigned int Q_row, unsigned int Q_col,
    unsigned int nbiters)
{
    assert(P_col == Q_col && P_col == 3);
    extern __shared__ double s_data[]; // first 3 * sizeof(double) bytes are used to store p_point[0,1,2], then we have 1025 ICPCorresps
    ICPCorresp *s_min_point = (ICPCorresp*)(s_data + 3); // last is final min
    ICPCorresp *s_corresp = s_min_point + 1; // pointer to ICPCorresp structure
    unsigned int qid = threadIdx.x; //blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pid = blockIdx.x; // * blockDim.x + threadIdx.y; // blockDim.y == 1 and blockDim.x == 1024
    double dist, tmp;
    const double *q_point;
    const double *p_point = d_P + pid * P_col;
    unsigned int iter = 0;
    if (qid == 0)
    {
        // storing p_point x, y and z for all threads in block
        s_data[0] = p_point[0];
        s_data[1] = p_point[1];
        s_data[2] = p_point[2];
        s_min_point[0] = { DBL_MAX, 0 };
    }
    __syncthreads(); // wait for x,y and z of p_point (should be done at first iter only)
    do {
        if (qid >= Q_row) {
            s_corresp[threadIdx.x] = { DBL_MAX, 0 };
            return;
        }
        q_point = d_Q + qid * Q_col;
        // compute distance for qid
        dist = 0;
        tmp = s_data[0] - q_point[0];
        dist += tmp * tmp;
        tmp = s_data[1] - q_point[1];
        dist += tmp * tmp;
        tmp = s_data[2] - q_point[2];
        dist += tmp * tmp;
        // store the distance
        s_corresp[threadIdx.x] = { dist, qid };
        __syncthreads();
        // min dist with friends
        assert(is_power_of_2(blockDim.x / 2));
        // TODO: loop unrolling
        for(unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride) {
                if (s_corresp[threadIdx.x].dist > s_corresp[threadIdx.x + stride].dist)
                    s_corresp[threadIdx.x] = s_corresp[threadIdx.x + stride];
            }
            __syncthreads();
        }
        // total min dist
        if (threadIdx.x == 0)
        {
            if (s_min_point[0].dist > s_corresp[0].dist)
                s_min_point[0] = s_corresp[0];
        }
        // increase qid
        qid += blockDim.x; // += 1024
    } while(++iter < nbiters);
    if (threadIdx.x == 0)
        d_array_correspondances[pid] = s_min_point[0].id;
}

__host__ void get_array_correspondences_optimized(unsigned int* d_array_correspondances, const double *d_P, const double *d_Q,
    unsigned int P_row, unsigned int P_col, unsigned int Q_row, unsigned int Q_col)
{
    dim3 gridsize(P_row,1);
    dim3 blocksize(1024,1);
    unsigned int nbiters = std::ceil((float)Q_row / blocksize.x);

    std::cerr << std::endl << "gridsize.x: " << gridsize.x << std::endl;
    std::cerr << "blocksize.x: " << blocksize.x << std::endl;
    std::cerr << "nbiters:" << nbiters << std::endl;

    get_array_correspondences_optimized_kernel<<<gridsize, blocksize, sizeof(ICPCorresp) * blocksize.x + sizeof(ICPCorresp) + 3 * sizeof(double)>>>(d_array_correspondances,
        d_P, d_Q, P_row, P_col, Q_row, Q_col, nbiters);
    cudaDeviceSynchronize();
    cudaCheckError();
}
