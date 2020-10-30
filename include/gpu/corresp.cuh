#pragma once

typedef struct
{
    double dist;
    unsigned int id;
} ICPCorresp;

/**
 ** \brief get_correspondences_kernel is a destructive in-place min distance axis=1 reduction kernel
 ** it should be called with <<<gridsize, blocksize, blockdim.x * sizeof(double)>>>
 ** must be called with gridsize.y = dist_0 and gridsize.x == 1, since this is an inplace destructive operation
 ** blocksize.x == dist_1 and blocksize.y == 1
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
 **/
__host__ void get_correspondences(ICPCorresp *d_dist,
    size_t dist_pitch, size_t dist_0, size_t dist_1, bool sync);
