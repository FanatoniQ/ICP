#pragma once

#include "libgpuicp/dist.cuh"
#include "libgpuicp/corresp.cuh"
#include "libgpuicp/crosscov.cuh"

/**
 ** \brief batch computation of distances, correspondences and cross-covariances.
 ** will perform Plines / batchsize iterations to finally fill d_cov with final cross-covariance for P and Q
 ** \note data should be allocated using proper batchsize
 **/
 __host__ void get_batch_cov(float *d_P, size_t Plines, size_t Pcols, size_t p_pitch,
    float *d_Q, size_t Qlines, size_t Qcols, size_t q_pitch,
    ICPCorresp *d_dist, size_t dist_0, size_t dist_1, size_t dist_pitch,
    float *d_R, size_t Rlines, size_t Rcols, size_t r_pitch,
    float *d_cov, size_t covLines, size_t covCols, size_t cov_pitch,
    size_t batchsize);