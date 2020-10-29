#include <limits>

struct Correps
{
    double dist;
    unsigned int id;
};

/**
 ** \brief get_correspondences_kernel is a destructive in-place min distance axis=1 reduction kernel
 ** it should be called with <<<gridsize, blocksize, blockdim.x * sizeof(double)>>>
 ** must be called with gridsize.y = dist_0 and gridsize.x == 1, since this is an inplace destructive operation
 ** blocksize.x == dist_1 and blocksize.y == 1
 ** each block treats a line min reduction using shared memory and stores the result at its line start
 **
 ** \param dist_pitch pitch of d_dist IN bytes
 **/
__global__ void get_correspondences_kernel(const struct Correps *d_dist,
    int dist_pitch, size_t dist_0, size_t dist_1)
{
    extern __shared__ struct Correps s_data[]; // s_data is of size blockDim.x
    int threadid = threadIdx.x; // thread id in the block
    int lineid = blockIdx.y; // line
    int dataid = blockIdx.x * blockDim.x + threadIdx.x; // column
    if (dataid >= dist_1 || lineid >= dist_0) {
        s_data[threadid] = { DBL_MAX,dataid };
        return;
    }
    // each thread copies to shared memory
    struct Correps *d_distline = (struct Correps *)((char *)d_dist + lineid * pitch);
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
        d_distline[blockIdx.x] = s_data[0];
}