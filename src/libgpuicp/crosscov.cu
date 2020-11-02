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
    //printf("%u -> %u\n", idp, idq);
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

    dim3 blocksize(1024, 1); // TODO: change nb threads
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

__device__ void increment_cov(double *d_cov, double *d_pline, double *d_qline)
{
    /*
    d_cov[0] = d_cov[0] + d_qline[0] * d_pline[0];
    d_cov[1] = d_cov[1] + d_qline[0] * d_pline[1];
    d_cov[2] = d_cov[2] + d_qline[0] * d_pline[2];

    d_cov[3] = d_cov[3] + d_qline[1] * d_pline[0];
    d_cov[4] = d_cov[4] + d_qline[1] * d_pline[1];
    d_cov[5] = d_cov[5] + d_qline[1] * d_pline[2];

    d_cov[6] = d_cov[6] + d_qline[2] * d_pline[0];
    d_cov[7] = d_cov[7] + d_qline[2] * d_pline[1];
    d_cov[8] = d_cov[8] + d_qline[2] * d_pline[2];
    */
    atomicAdd(d_cov, d_qline[0] * d_pline[0]);
    atomicAdd(d_cov + 1, d_qline[0] * d_pline[1]);
    atomicAdd(d_cov + 2, d_qline[0] * d_pline[2]);

    atomicAdd(d_cov + 3, d_qline[1] * d_pline[0]);
    atomicAdd(d_cov + 4, d_qline[1] * d_pline[1]);
    atomicAdd(d_cov + 5, d_qline[1] * d_pline[2]);

    atomicAdd(d_cov + 6, d_qline[2] * d_pline[0]);
    atomicAdd(d_cov + 7, d_qline[2] * d_pline[1]);
    atomicAdd(d_cov + 8, d_qline[2] * d_pline[2]);

}

__global__ void get_array_cross_cov_kernel(double * d_cov, unsigned int* d_array_correspondances, double *d_P, double *d_Q,
    unsigned int P_row, unsigned int P_col, unsigned int Q_row, unsigned int Q_col)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= P_row)
        return;
    
    auto i = index;
    auto j = d_array_correspondances[index];
    //printf("i vaut %d et j vaut %d\n", i, j);
    double *d_ppoint = d_P + i * P_col;
    double *d_qpoint = d_Q + j * Q_col;

    //printf("%lf et %lf", *d_ppoint, *d_qpoint);
    //for (int i = 0; i < 3; i++)
    //   printf("%lf and %lf\n", d_ppoint[i], d_qpoint[i]);

    increment_cov(d_cov, d_ppoint, d_qpoint);

    //for (int i = 0; i < 9; i++)
    //    printf(" %lf \n", d_cov[i]);
}

__host__ void get_array_cross_cov(double* d_cov, unsigned int* d_array_correspondances, double *d_P, double *d_Q,
    unsigned int P_row, unsigned int P_col, unsigned int Q_row, unsigned int Q_col)
{
    runtime_assert(P_col == Q_col && Q_col == 3, "Only works in dim 3 !");

    dim3 blocksize(1024, 1);
    dim3 gridsize(std::ceil((float)P_row / blocksize.x), 1);
    std::cerr << std::endl << "gridsize.x: " << gridsize.x << std::endl;
    std::cerr << "blocksize.x: " << blocksize.x << std::endl;
    
    cudaMemset(d_cov, 0, sizeof(double) * P_col * Q_col);

    get_array_cross_cov_kernel<<<gridsize, blocksize>>>(d_cov, d_array_correspondances, d_P, d_Q, P_row, P_col, Q_row, Q_col);
    cudaDeviceSynchronize();
    cudaCheckError();
}