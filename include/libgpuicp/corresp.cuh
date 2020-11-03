#pragma once

typedef struct
{
    double dist;
    unsigned int id;
} ICPCorresp;

/**
 ** \brief get_correspondences_kernel is a destructive in-place min distance axis=1 reduction kernel
 ** it should be called with <<<gridsize, blocksize, blockdim.x * sizeof(double)>>>
 ** must be called with gridsize.y = dist_0, since this is an inplace destructive operation
 ** blocksize.x * gridsize.x >= dist_1 and blocksize.y == 1. blocksize.x must be a power of 2, 1024 is a good choice
 ** each block treats a line min reduction using shared memory and stores the result at its line start
 **
 ** \param d_dist the struct Correps matrix to be in place min reduced
 ** \param dist_pitch pitch of d_dist IN bytes
 ** \param dist_0 the number of lines in d_dist (== gridsize.y)
 ** \param dist_1 the number of columns in d_dist (== blocksize.x)
 **/
__global__ void get_correspondences_kernel(ICPCorresp *d_dist,
    size_t dist_pitch, size_t dist_0, size_t dist_1);

/**
 ** \brief get_correspondences wrapper around get_correspondences_kernel
 **
 ** \param d_dist the ICPCorresp struct matrix to be in place min reduced
 ** \param dist_pitch pitch of d_dist IN bytes
 ** \param dist_0 the number of lines in d_dist (== gridsize.y)
 ** \param dist_1 the number of columns in d_dist (== blocksize.x)
 ** \param sync wether to wait for device to finish or not
 ** \param threads minimum number of threads (rounded to next power of two)
 **/
__host__ void get_correspondences(ICPCorresp *d_dist,
    size_t dist_pitch, size_t dist_0, size_t dist_1, bool sync = true, size_t threads = 1024);

/**
 ** \brief get_array_correspondences wrapper around get_array_correspondences_kernel
 ** P and Q and stores them in d_array_correspondances
 **
 ** \param d_array_correspondances the resulting array
 ** \param P the first matrix
 ** \param Q the second matrix
 ** \param P_row the number of rows in P
 ** \param P_col the number of columns in P
 ** \param Q_row the number of rows in Q
 ** \param Q_col the number of columns in Q
 **/
__host__ void get_array_correspondences(unsigned int* d_array_correspondances, double *d_P, double *d_Q,
    unsigned int P_row, unsigned int P_col, unsigned int Q_row, unsigned int Q_col,
    size_t p_pitch, size_t q_pitch);


/**
 ** \brief get_correspondences_kernel computes correspondances between
 **
 ** \param d_array_correspondances the resulting array
 ** \param P the first matrix
 ** \param Q the second matrix
 ** \param P_row the number of rows in P
 ** \param P_col the number of columns in P
 ** \param Q_row the number of rows in Q
 ** \param Q_col the number of columns in Q
 **/
__global__ void get_array_correspondences_kernel(unsigned int *d_array_correspondances, double *d_P, double *d_Q, unsigned int P_row, unsigned int P_col, unsigned int Q_row, unsigned int Q_col,
    size_t p_pitch, size_t q_pitch);
