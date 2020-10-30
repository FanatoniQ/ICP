#include <limits>
#include <float.h>
#include <assert.h>
#include <stdio.h>
#include <iostream>

#include "libgpualg/mean.cuh"
#include "libgpuicp/crosscov.cuh"
#include "error.cuh"
#include "error.hpp"

__global__ void get_cross_cov_kernel(const double *d_P, const double *d_Q, double *d_R, const ICPCorresp *d_dist,
    size_t p_0, size_t p_1, size_t p_pitch,
    size_t q_0, size_t q_1, size_t q_pitch,
    size_t r_0, size_t r_1, size_t r_pitch,
    size_t dist_0, size_t dist_1, size_t dist_pitch)
{
    assert(blockIdx.y == 0 && threadIdx.y == 0); // we are in 1d
    //unsigned int colid = blockIdx.y * blockDim.y + threadIdx.y; // y index (unused)
    unsigned int lineid = blockIdx.x * blockDim.x + threadIdx.x; // P index
    if (lineid >= dist_0) {
        return;
    }
    ICPCorresp *d_distline = (ICPCorresp *)((char*)d_dist + lineid * dist_pitch);
    unsigned int idp = lineid; // p_point id
    unsigned int idq = d_distline[0].id; // q_point id
    // if pitch not in bytes:
    //unsigned int idq = d_dist[lineid * dist_pitch].id;//d_distline[0].id; // q_point id

    double *d_rline = (double *)((char *)d_R + idp * r_pitch);
    double *d_pline = (double *)((char *)d_P + idp * p_pitch);
    double *d_qline = (double *)((char *)d_Q + idq * q_pitch);
    /**
    // generic case:
    //fake transpose dot between P and Q:
    //d_rline stores P.dot(Q.T) flattened in the line
    for (size_t q_i = 0; q_i < q_1 ; ++q_i)
    {
        for (size_t p_i = 0; p_i < p_1 ; ++p_i)
        {
            d_rline[p_i * q_1 + p_j] = d_qline[q_i] * d_pline[p_i];
        }
    }
    **/
    printf("%u -> %u\n", idp, idq);
    //printf("P{%u:} %lf %lf %lf, Q{%u:} %lf %lf %lf \n", idp, d_pline[0], d_pline[1], d_pline[2], idq, d_qline[0], d_qline[1], d_qline[2]);
    // loop unrolled version for dim3:
    assert(p_1 == q_1 && p_1 == 3 && "Invalid: only dim3 is supported !");
    d_rline[0] = d_qline[0] * d_pline[0];
    d_rline[1] = d_qline[0] * d_pline[1];
    d_rline[2] = d_qline[0] * d_pline[2];

    d_rline[3] = d_qline[1] * d_pline[0];
    d_rline[4] = d_qline[1] * d_pline[1];
    d_rline[5] = d_qline[1] * d_pline[2];

    d_rline[6] = d_qline[2] * d_pline[0];
    d_rline[7] = d_qline[2] * d_pline[1];
    d_rline[8] = d_qline[2] * d_pline[2];
    //printf("%u %u: %lf %lf %lf | %lf %lf %lf | %lf %lf %lf\n", idp, idq, d_rline[0], d_rline[1], d_rline[2], d_rline[3], d_rline[4], d_rline[5], d_rline[6], d_rline[7], d_rline[8]);
    //printf("P{%u:} %lf %lf %lf, Q{%u:} %lf %lf %lf \n", idp, d_pline[0], d_pline[1], d_pline[2], idq, d_qline[0], d_qline[1], d_qline[2]);
}

__host__ void get_cross_cov(const double *d_P, const double *d_Q, double **d_R, const ICPCorresp *d_dist,
    size_t p_0, size_t p_1, size_t p_pitch,
    size_t q_0, size_t q_1, size_t q_pitch,
    size_t r_0, size_t r_1, size_t *r_pitch,
    size_t dist_0, size_t dist_1, size_t dist_pitch, bool sync)
{
    runtime_assert(p_1 == q_1 && dist_1 == q_0 && dist_0 == p_0, "Invalid shapes !");
    if (*d_R == nullptr)
    {
        cudaMallocPitch(d_R, r_pitch, p_1 * q_1 * sizeof(double), dist_0);
        cudaCheckError();
    }

    dim3 blocksize(8, 1); // TODO: change nb threads
    dim3 gridsize(std::ceil((float)dist_0 / blocksize.x), 1);
    std::cerr << blocksize.x << " èè " << gridsize.x << std::endl;
    
    get_cross_cov_kernel<<<gridsize, blocksize>>>(d_P, d_Q, *d_R, d_dist,
        p_0, p_1, p_pitch,
        q_0, q_1, q_pitch,
        r_0, r_1, *r_pitch,
        dist_0, dist_1, dist_pitch);
    
    if (sync) {
        cudaDeviceSynchronize();
        cudaCheckError();
    }
}
