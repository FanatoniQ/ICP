#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <iostream>
#include <limits>
#include <float.h>
#include <assert.h>

// CPU
#include "libCSV/csv.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/CPUView.hpp"
#include "libalg/alg.hpp"
#include "libalg/print.hpp"
#include "error.hpp"

// GPU
#include "libgpualg/ope.cuh"
#include "libgpualg/mean.cuh"
#include "error.cuh"
#include "libgpuicp/dist.cuh"
#include "libgpuicp/corresp.cuh"
#include "libgpuicp/crosscov.cuh"
#include "libgpuicp/batchcovs.cuh"

int main(int argc, char **argv)
{
    runtime_assert(argc == 3 || argc == 4, "./batchcov file1 file2 [batchsize]");
    std::string f1Header{};
    size_t Qlines, Qcols, Plines, Pcols;
    double *Pt = readCSV(argv[1], f1Header, Plines, Pcols);
    double *Qt = readCSV(argv[2], f1Header, Qlines, Qcols);

    size_t batchsize = 16;
    if (argc == 4)
        batchsize = std::stol(argv[3]);

    // device P matrix
    size_t p_pitch = Pcols * sizeof(double);
    double *d_P;
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
    size_t dist_0 = batchsize, dist_1 = Qlines;
    size_t dist_pitch;
    ICPCorresp *d_dist;
    cudaMallocPitch((void **)&d_dist, &dist_pitch, dist_1 * sizeof(ICPCorresp), dist_0);
    cudaCheckError();

    // device cross-covs flattened
    size_t Rlines = batchsize, Rcols = Pcols * Qcols;
    size_t r_pitch = Rcols * sizeof(double);
    double *d_R;
    // or 2d...
    cudaMalloc((void**)&d_R, Rlines * r_pitch);
    cudaCheckError();

    // device FINAL cross-cov (flattened)
    size_t covLines = Qcols, covCols = Pcols;
    size_t cov_pitch = covCols * covLines * sizeof(double);
    double *d_cov;
    cudaMalloc((void**)&d_cov, 1 * cov_pitch);
    cudaCheckError();

    // CALL get_batch_cov
    get_batch_cov(d_P, Plines, Pcols, p_pitch,
        d_Q, Qlines, Qcols, q_pitch,
        d_dist, dist_0, dist_1, dist_pitch,
        d_R, Rlines, Rcols, r_pitch,
        d_cov, covLines, covCols, cov_pitch,
	batchsize
    );

    double *h_cov = (double *)malloc(covLines * covCols * sizeof(double));
    cudaMemcpy(h_cov, d_cov, covLines * covCols * sizeof(double), cudaMemcpyDeviceToHost);
    auto FULLGPUCOV = CPUMatrix(h_cov, covLines, covCols);
    std::cout << FULLGPUCOV << std::endl;

    cudaFree(d_cov);
    cudaCheckError();
    cudaFree(d_P);
    cudaCheckError();
    cudaFree(d_Q);
    cudaCheckError();
    cudaFree(d_R);
    cudaCheckError();
    cudaFree(d_dist);
    cudaCheckError();
}
