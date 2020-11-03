#pragma once

#include "libgpuicp/corresp.cuh"

/**
 ** \brief optimized version of get_array_correspondences_kernel, we have one block per point in p, 1024 threads blocks
 ** which loops if more than 1024 points in q. ICPCorresp structure is used for storing dist,idq, this kernel min reduces
 ** in place this array and repeats the process
 ** we launch with gridsize.x == p_0, gridsize.y == 1; blocksize.x == 1024, blocksize.y == 1
 ** <<<dim3(p_0,1), dim3(1024,1), sizeof(ICPCorresp) * 1024 + sizeof(ICPCorresp) + 3 * sizeof(float)>>>
 ** \note C.C. >=3.0 => maximum 2^31 - 1 blocks on x dimension and 65535 on y and z
 ** \note blockIdx.x is used as P index
 ** \note threadIdx.x is not related to blockIdx.x, it is a threadid used for Q index
 ** \note we loop in case blockDim.x is not enough
 ** \note we do multiple iterations with nbiters since we want to share the data between threads and don't want to
 ** write to global memory
 ** \todo we could compute cross-cov here to be even quicker
 **/
__global__ void get_array_correspondences_optimized_kernel(unsigned int *d_array_correspondances,
    const float *d_P, const float *d_Q, unsigned int P_row, unsigned int P_col, unsigned int Q_row, unsigned int Q_col,
    unsigned int nbiters);

__host__ void get_array_correspondences_optimized(unsigned int* d_array_correspondances, const float *d_P, const float *d_Q,
    unsigned int P_row, unsigned int P_col, unsigned int Q_row, unsigned int Q_col);

 //////////////
// ONE ITER //
/////////////

__global__ void get_array_correspondences_optimized_one_iter_kernel(ICPCorresp *d_dists, unsigned int dist_1,
    const float *d_P, const float *d_Q, unsigned int P_row, unsigned int P_col, unsigned int Q_row, unsigned int Q_col);

__global__ void get_array_reduced_correspondences_kernel(unsigned int *d_array_correspondances, ICPCorresp *d_dist,
    size_t dist_pitch, size_t dist_0, size_t dist_1);

__host__ void get_array_correspondences_optimized_one_iter(unsigned int *d_array_correspondances,
    ICPCorresp **d_dist, unsigned int *dist_1, const float *d_P, const float *d_Q,
    unsigned int P_row, unsigned int P_col, unsigned int Q_row, unsigned int Q_col);
