#pragma once

#include "gpu/corresp.cuh"
#include "gpu/dist.cuh"

/**
 ** \brief get_cross_cov_kernel computes the cross covariance matrix, which is stored as a flattened array
 ** at each line of d_R.
 ** This kernel should be called using 1D blocks and threads <<<gridsize, blocksize>>> with:
 ** - gridsize.x * blocksize.x >= dist_0 (== p_0)
 ** - gridsize.y == 1 and blocksize.y == 1
 **
 ** \param d_P the P matrix of shape p_0,p_1
 ** \param d_Q the Q matrix of shape q_0,q_1
 ** \param d_R the resulting R matrix os shape p_0,p_1*q_1 (p_1 == q_1 == 3)
 ** \param d_dist the reduced correspondances/distances matrix between P and Q, result from get_correspondences_kernel
 ** with shape p_0,1 and the given pitch
 ** \param p_0 number of lines in P
 ** \param p_1 number of columns in P
 ** \param p_pitch pitch of P IN bytes
 ** \param q_0 number of lines in Q
 ** \param q_1 number of columns in Q
 ** \param q_pitch pitch of Q IN bytes
 ** \param r_0 number of lines in the resulting matrix
 ** \param r_1 number of columns in the resulting matrix
 ** \param r_pitch pitch of the resulting matrix IN bytes
 ** \param dist_0 number of lines in d_dist (must be == p_0)
 ** \param dist_1 number of columns in d_dist (must be == q_0)
 ** \param dist_pitch pointer to the pitch of d_dist IN bytes
 **/
 __global__ void get_cross_cov_kernel(const double *d_P, const double *d_Q, double *d_R, const ICPCorresp *d_dist,
    size_t p_0, size_t p_1, size_t p_pitch,
    size_t q_0, size_t q_1, size_t q_pitch,
    size_t r_0, size_t r_1, size_t r_pitch,
    size_t dist_0, size_t dist_1, size_t dist_pitch);

/**
 ** \brief get_cross_cov wrapper around get_cross_cov_kernel
 **
 ** \param d_P the P matrix of shape p_0,p_1
 ** \param d_Q the Q matrix of shape q_0,q_1
 ** \param d_R pointer to the resulting R matrix os shape p_0,p_1*q_1 (p_1 == q_1 == 3), alloc if is nullptr
 ** \param d_dist the reduced correspondances/distances matrix between P and Q, result from get_correspondences_kernel
 ** with shape p_0,1 and the given pitch
 ** \param p_0 number of lines in P
 ** \param p_1 number of columns in P
 ** \param p_pitch pitch of P IN bytes
 ** \param q_0 number of lines in Q
 ** \param q_1 number of columns in Q
 ** \param q_pitch pitch of Q IN bytes
 ** \param r_0 number of lines in the resulting matrix
 ** \param r_1 number of columns in the resulting matrix
 ** \param r_pitch pointer to the pitch of the resulting matrix IN bytes
 ** \param dist_0 number of lines in d_dist (must be == p_0)
 ** \param dist_1 number of columns in d_dist (must be == q_0)
 ** \param dist_pitch pointer to the pitch of d_dist IN bytes
 ** \param sync wether to wait
 **/
__host__ void get_cross_cov(const double *d_P, const double *d_Q, double **d_R, const ICPCorresp *d_dist,
    size_t p_0, size_t p_1, size_t p_pitch,
    size_t q_0, size_t q_1, size_t q_pitch,
    size_t r_0, size_t r_1, size_t *r_pitch,
    size_t dist_0, size_t dist_1, size_t dist_pitch, bool sync);