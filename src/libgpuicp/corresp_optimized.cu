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
    extern __shared__ double s_corresp[]; // improvement: no bank conflict when splitting ICPCorresp
    
    double *s_dists = s_corresp;
    unsigned int *s_ids = (unsigned int *)(s_dists + blockDim.x);
    double *s_min_dist = (double *)(s_ids + blockDim.x);
    unsigned int *s_min_id = (unsigned int *)(s_min_dist + 1);

    unsigned int qid = threadIdx.x; //blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pid = blockIdx.x; // * blockDim.x + threadIdx.y; // blockDim.y == 1 and blockDim.x == 1024
    double dist, tmp;
    const double *q_point;
    const double *p_point = d_P + pid * P_col;
    unsigned int iter = 0;
    if (qid == 0)
    {
	s_min_dist[0] = DBL_MAX;
    }
    __syncthreads(); // wait for x,y and z of p_point (should be done at first iter only)
    do {
        if (qid >= Q_row) {
	    s_dists[threadIdx.x] = DBL_MAX;
            return;
        }
        q_point = d_Q + qid * Q_col;
        // compute distance for qid
        dist = 0;
        tmp = p_point[0] - q_point[0];
        dist += tmp * tmp;
        tmp = p_point[1] - q_point[1];
        dist += tmp * tmp;
        tmp = p_point[2] - q_point[2];
        dist += tmp * tmp;
        // store the distance
	s_dists[threadIdx.x] = dist;
	s_ids[threadIdx.x] = qid;
        __syncthreads();
        // min dist with friends
        assert(is_power_of_2(blockDim.x / 2));
        // TODO: loop unrolling
        for(unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride) {
		if (s_dists[threadIdx.x] > s_dists[threadIdx.x + stride])
		{
s_dists[threadIdx.x] = s_dists[threadIdx.x + stride];
s_ids[threadIdx.x] = s_ids[threadIdx.x + stride];
		}
            }
            __syncthreads();
        }
        // total min dist
        if (threadIdx.x == 0)
        {
	    if (s_min_dist[0] > s_dists[0]) {
		s_min_dist[0] = s_dists[0];
		s_min_id[0] = s_ids[0];
	    } 
        }
        // increase qid
        qid += blockDim.x; // += 1024
    } while(++iter < nbiters);
    if (threadIdx.x == 0)
        d_array_correspondances[pid] = s_min_id[0];//.id;
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

    get_array_correspondences_optimized_kernel<<<gridsize, blocksize, sizeof(ICPCorresp) * blocksize.x + sizeof(ICPCorresp)>>>(d_array_correspondances,
        d_P, d_Q, P_row, P_col, Q_row, Q_col, nbiters);
    cudaDeviceSynchronize();
    cudaCheckError();
}

/**
 ** \brief optimized version of get_array_correspondences_kernel, we have one block per point in p, 1024 threads blocks
 ** which loops if more than 1024 points in q. ICPCorresp structure is used for storing dist,idq, this kernel min reduces
 ** in place this array and repeats the process
 ** we launch with gridsize.x == nbblocksPerLine, gridsize.y == p_0; blocksize.x == 1024, blocksize.y == 1
 ** nbblocksPerLine == std::ceil((float)q_0 / blocksize.x)
 ** <<<dim3(nbblocksPerLine,p_0), dim3(1024,1), sizeof(ICPCorresp) * 1024 + 3 * sizeof(double)>>>
 ** \note C.C. >=3.0 => maximum 2^31 - 1 blocks on x dimension and 65535 on y and z, maybe swap gridsize.y usage
 ** \note x axis is used as Q index
 ** \note gridsize.y == p_0
 **/
__global__ void get_array_correspondences_optimized_one_iter_kernel(ICPCorresp *d_dists, unsigned int dist_1,
const double *d_P, const double *d_Q, unsigned int P_row, unsigned int P_col, unsigned int Q_row, unsigned int Q_col)
{
    assert(P_col == Q_col && P_col == 3);
    extern __shared__ double s_dists[];
    unsigned int *s_ids = (unsigned int *)(s_dists + blockDim.x); // we could use short unsigned int
    unsigned int qid = blockIdx.x * blockDim.x + threadIdx.x; // each x represent a Q point
    unsigned int pid = blockIdx.y; // gridsize.y is the number of points in P
    double dist, tmp;
    const double *q_point;
    const double *p_point = d_P + pid * P_col;
    if (qid >= Q_row) {
	s_dists[threadIdx.x] = DBL_MAX;
	s_ids[threadIdx.x] = 0;
        return;
    }
    q_point = d_Q + qid * Q_col;
    // compute distance for qid
    dist = 0;
    tmp =  p_point[0] - q_point[0];
    dist += tmp * tmp;
    tmp =  p_point[1] - q_point[1];
    dist += tmp * tmp;
    tmp =  p_point[2] - q_point[2];
    dist += tmp * tmp;
    // store the distance
    s_dists[threadIdx.x] = dist;
    s_ids[threadIdx.x] = qid;
    __syncthreads();
    // min dist with friends
    assert(is_power_of_2(blockDim.x / 2));
    // TODO: loop unrolling
    for(unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride) {
            if (s_dists[threadIdx.x] > s_dists[threadIdx.x + stride]) {
                 s_dists[threadIdx.x] = s_dists[threadIdx.x + stride];
                 s_ids[threadIdx.x] = s_ids[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        d_dists[pid * dist_1 + blockIdx.x] = { s_dists[0], s_ids[0] }; // partial reduce
}

__global__ void get_array_reduced_correspondences_kernel(unsigned int *d_array_correspondances, ICPCorresp *d_dist,
    size_t dist_pitch, size_t dist_0, size_t dist_1)
{
    extern __shared__ double s_reducedists[]; // s_data is of size blockDim.x
    unsigned int *s_reduceids = (unsigned int *)(s_reducedists + blockDim.x);
    unsigned int threadid = threadIdx.x; // thread id in the block
    unsigned int lineid = blockIdx.y; // line
    unsigned int dataid = threadIdx.x; // only used in final reduce //blockIdx.x * blockDim.x + threadIdx.x; // column
    if (dataid >= dist_1 || lineid >= dist_0) {
	s_reducedists[threadid] = DBL_MAX;
        return;
    }
    ICPCorresp *d_distline = (ICPCorresp *)((char *)d_dist + lineid * dist_pitch);
    s_reducedists[threadid] = d_distline[dataid].dist;
    s_reduceids[threadid] = d_distline[dataid].id;
    __syncthreads();
    assert(is_power_of_2(blockDim.x / 2)); // if not power of 2 ...
    for (size_t stride = blockDim.x / 2; stride > 0; stride = stride >> 1)
    {
        if (threadid < stride) { // a lot of threads are idle...
	    if (s_reducedists[threadid] > s_reducedists[threadid + stride]) {
		    s_reducedists[threadid] = s_reducedists[threadid + stride];
		    s_reduceids[threadid] = s_reduceids[threadid + stride];
	    }
        }
        __syncthreads();
    }
    if (threadid == 0)
        d_array_correspondances[lineid] = s_reduceids[0];
}

__host__ void get_array_correspondences_optimized_one_iter(unsigned int *d_array_correspondances,
    ICPCorresp **d_dist, unsigned int *dist_1, const double *d_P, const double *d_Q,
    unsigned int P_row, unsigned int P_col, unsigned int Q_row, unsigned int Q_col)
{
    dim3 blocksize(1024,1); // FIXME: changing this from 1024 to 4 got rid of cross_cov mem access error
    unsigned int nbblocksPerLine = std::ceil((float)Q_row / blocksize.x);
    dim3 gridsize(nbblocksPerLine,P_row); // \todo: maybe swap (see above)

    if (*d_dist == nullptr)
    {
        *dist_1 = nbblocksPerLine;
        cudaMalloc(d_dist, *dist_1 * P_row * sizeof(ICPCorresp));
        cudaCheckError();
    }
    runtime_assert(*dist_1 == nbblocksPerLine, "invalide d_dist shape !");

    std::cerr << std::endl << "gridsize.x: " << gridsize.x << std::endl;
    std::cerr << "blocksize.x (nbblocksPerLine): " << blocksize.x << " blocksize.y:" << blocksize.y << std::endl;

    get_array_correspondences_optimized_one_iter_kernel<<<gridsize, blocksize, sizeof(ICPCorresp) * blocksize.x>>>(*d_dist,
        *dist_1,
        d_P, d_Q,
        P_row, P_col, Q_row, Q_col);
    cudaDeviceSynchronize();
    cudaCheckError();

    blocksize = dim3(1024, 1);
    gridsize = dim3(1, P_row);
    
    std::cerr << "nbthreads: " << blocksize.x << " nblines: " << gridsize.y << " nbblocksPerLine: " << gridsize.x << std::endl;
    get_array_reduced_correspondences_kernel<<<gridsize, blocksize, blocksize.x * sizeof(ICPCorresp)>>>(d_array_correspondances, *d_dist, sizeof(ICPCorresp) * nbblocksPerLine, P_row, nbblocksPerLine);
    cudaDeviceSynchronize();
    cudaCheckError();
}
