#pragma once

#include "libgpuicp/corresp.cuh"

/**
 ** \brief get_distances_kernel fills the d_dist ICPCorresp matrix
 ** each thread will compute euclidian distances between two points (one line of P and one line of Q)
 ** get_distances_kernel should be launched with <<<gridsize, blocksize>>> such as:
 ** - gridsize.x * blocksize.x >= dist_1
 ** - dist_1 == q_0
 ** - gridsize.x * blocksize.x >= dist_0
 ** - dist_0 == p_0
 ** - p_1 == q_1
 ** pitches are IN bytes
 **/
 __global__ void get_distances_kernel(const float *d_P, const float *d_Q, ICPCorresp *d_dist,
    size_t p_0, size_t p_1, size_t p_pitch,
    size_t q_0, size_t q_1, size_t q_pitch,
    size_t dist_0, size_t dist_1, size_t dist_pitch);

/**
 ** \brief get_distances wrapper around get_distances_kernel
 **
 ** \param d_P the P device matrix of shape p_0, p_1
 ** \param d_Q the Q device matrix of shape q_0, q_1
 ** \param d_dist the pointer to the resulting distance correspondance matrix d_dist if nullptr we allocate
 ** \param p_0 number of lines in P
 ** \param p_1 number of columns in P
 ** \param p_pitch pitch of P IN bytes
 ** \param q_0 number of lines in Q
 ** \param q_1 number of columns in Q
 ** \param q_pitch pitch of Q IN bytes
 ** \param dist_0 number of lines in d_dist (must be == p_0)
 ** \param dist_1 number of columns in d_dist (must be == q_0)
 ** \param dist_pitch pointer to the pitch of d_dist IN bytes
 ** \param sync wether to wait for kernel or not
 **/
__host__ void get_distances(const float *d_P, const float *d_Q, ICPCorresp **d_dist,
    size_t p_0, size_t p_1, size_t p_pitch,
    size_t q_0, size_t q_1, size_t q_pitch,
    size_t dist_0, size_t dist_1, size_t *dist_pitch, bool sync);
