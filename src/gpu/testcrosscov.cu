#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <iostream>
#include <limits>
#include <float.h>

// CPU
#include "libCSV/csv.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/alg.hpp"
#include "libalg/print.hpp"
#include "error.hpp"


// GPU
#include "libgpualg/mean.cuh"
#include "error.cuh"
#include "gpu/icp.cuh"
#include "gpu/corresp.cuh"
#include "gpu/dist.cuh"
#include "gpu/corresp.cuh"

__host__ double *get_cross_covs_cpu(const CPUMatrix &P, size_t p_0, size_t p_1,
    const CPUMatrix &Q, size_t q_0, size_t q_1,
    ICPCorresp *d_dist, size_t dist_0, size_t dist_1, size_t dist_pitch)
{
    size_t ref_pitch = q_1 * p_1 * sizeof(double);
    double *h_ref = (double*)malloc(p_0 * ref_pitch);

    size_t h_dist_pitch = dist_1 * sizeof(ICPCorresp);
    ICPCorresp *h_dist = (double*)malloc(dist_0 * h_dist_pitch);

    cudaMemcpy2D(h_dist, h_dist_pitch, d_dist, dist_pitch, dist_1 * sizeof(ICPCorresp), dist_0, cudaMemcpyDeviceToHost);
    cudaCheckError();

    for (size_t i; i < p_0; ++i)
    {
        size_t idq = h_dist[i].id;
        auto cov = Q.getLine(idq).dot(P.getLine(i).transpose());
        assert(ref_pitch == cov.getDim0() * cov.getDim1());
        memcpy(h_ref + i * ref_pitch, cov.getArray(), cov.getDim0() * cov.getDim1() * sizeof(double));
    }

    return h_ref;
}

int main(int argc, char **argv)
{
    std::string f1Header{};
    size_t Qlines, Qcols, Plines, Pcols;
    double *Pt = readCSV(argv[1], f1Header, Plines, Pcols);
    CPUMatrix P = CPUMatrix(Pt, Plines, Pcols);
    double *Qt = readCSV(argv[2], f1Header, Qlines, Qcols);
    CPUMatrix Q = CPUMatrix(Qt, Qlines, Qcols);

    // device P matrix
    size_t p_pitch = Pcols * sizeof(double);
    double *d_P;
    //cudaMallodist_pitch((void **)&d_P, &p_pitch, Pcols * sizeof(double), Plines);
    cudaMalloc((void**)&d_P, Plines * p_pitch);
    cudaCheckError();
    cudaMemcpy2D(d_P, p_pitch, Pt, Pcols * sizeof(double), Pcols * sizeof(double), Plines, cudaMemcpyHostToDevice);
    cudaCheckError();

    // device Q matrix
    size_t q_pitch = Qcols * sizeof(double);
    double *d_Q;
    //cudaMallodist_pitch((void **)&d_Q, &q_pitch, Qcols * sizeof(double), Qlines);
    cudaMalloc((void**)&d_Q, Qlines * q_pitch);
    cudaCheckError();
    cudaMemcpy2D(d_Q, q_pitch, Qt, Qcols * sizeof(double), Qcols * sizeof(double), Qlines, cudaMemcpyHostToDevice);
    cudaCheckError();

    // device dist,id distance and corresp matrix
    size_t dist_0 = Plines, dist_1 = Qlines;
    size_t dist_pitch;
    ICPCorresp *d_dist;
    cudaMallodist_pitch((void **)&d_dist, &dist_pitch, dist_1 * sizeof(ICPCorresp), dist_0);
    cudaCheckError();
    //cudaMemcpy2D(d_dist, dist_pitch, C, Qlines * sizeof(ICPCorresp), Qlines * sizeof(ICPCorresp), Plines, cudaMemcpyHostToDevice);
    //cudaCheckError();

    // device cross-covs flattened
    size_t Rlines = Plines, Rcols = Pcols * Qcols;
    size_t r_pitch = Rcols * sizeof(double);
    double *d_R;
    // or 2d...
    cudaMalloc((void**)&d_R, Rlines * r_pitch);
    cudaCheckError();

    // DISTS
    get_distances(d_P, d_Q, &d_dist, Plines, Pcols, p_pitch, Qlines, Qcols, q_pitch, Plines, Qlines, &dist_pitch, true);
    std::cerr << "DISTS DONE" << std::endl;

    // CORRESPS
    get_correspondences(d_dist, dist_pitch, dist_0, dist_1, true);
    std::cerr << "CORRESPS DONE" << std::endl;

    /** Testing corresps: **/
    //ICPCorresp *h_res = (ICPCorresp *)malloc(Plines * Qlines * sizeof(ICPCorresp));
    //cudaMemcpy2D(h_res, Qlines * sizeof(ICPCorresp), d_dist, dist_pitch, 1 * sizeof(ICPCorresp), Plines, cudaMemcpyDeviceToHost);
    //cudaCheckError();

    // CROSS-COVS
    get_cross_cov(d_P, d_Q, &d_R, d_dist,
        Plines, Pcols, p_pitch,
        Qlines, Qcols, q_pitch,
        Rlines, Rcols, &r_pitch,
        dist_0, dist_1, dist_pitch, true);
    std::cerr << "CROSS-COVS DONE" << std::endl;

    /** Testing cross-covs: **/
    double *h_ref_cross_covs = get_cross_covs_cpu(P, Plines, Pcols, Q, Qlines, Qcols, d_dist, dist_0, dist_1, dist_pitch);
    double *h_r = (double*)malloc(Rlines * Rcols * sizeof(double));
    cudaMemcpy2D(h_r, Rcols * sizeof(double), d_dist, dist_pitch, Rcols * sizeof(double), Rlines, cudaMemcpyDeviceToHost);
    cudaCheckError();
    assert(memcmp(h_ref_cross_covs, h_r, Rlines * Rcols * sizeof(double)) == 0);
    
    free(h_r);
    free(h_ref_cross_covs);
    cudaFree(d_P);
    cudaCheckError();
    cudaFree(d_Q);
    cudaCheckError();
    cudaFree(d_R);
    cudaCheckError();
    cudaFree(d_dist);
    cudaCheckError();
}