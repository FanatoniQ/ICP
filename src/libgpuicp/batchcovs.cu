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

__host__ void get_batch_cov(double *d_P, size_t Plines, size_t Pcols, size_t p_pitch,
    double *d_Q, size_t Qlines, size_t Qcols, size_t q_pitch,
    ICPCorresp *d_dist, size_t dist_0, size_t dist_1, size_t dist_pitch,
    double *d_R, size_t Rlines, size_t Rcols, size_t r_pitch,
    double *d_cov, size_t covLines, size_t covCols, size_t cov_pitch,
    size_t batchsize)
{
    runtime_assert(dist_0 == batchsize, "Invalid d_dist shape for batch !");
    runtime_assert(Rlines == batchsize, "Invalid d_R shape for batch !");
    runtime_assert(covCols == 3 && covLines == 3, "only dim 3 points are supported !");
    cudaMemset(d_cov, 0, cov_pitch);
    cudaCheckError();
    size_t Pstartindex = 0;
    size_t nblines;

    while (Pstartindex < Plines)
    {
        nblines = MIN(Plines - Pstartindex, batchsize);

        // DISTS
        get_distances(d_P + Pstartindex * p_pitch / sizeof(double), d_Q, &d_dist, nblines, Pcols, p_pitch, Qlines, Qcols, q_pitch, nblines, dist_1, &dist_pitch, true);
        std::cerr << "DISTS DONE" << std::endl;

        // CORRESPS
        get_correspondences(d_dist, dist_pitch, nblines, dist_1, true);
        std::cerr << "CORRESPS DONE" << std::endl;

        // CROSS-COVS
        get_cross_cov(d_P + Pstartindex * p_pitch / sizeof(double), d_Q, &d_R, d_dist,
            nblines, Pcols, p_pitch,
            Qlines, Qcols, q_pitch,
            nblines, Rcols, &r_pitch,
            nblines, dist_1, dist_pitch, true);
        std::cerr << "CROSS-COVS DONE" << std::endl;

        // COVS SUM
        reduce_0(MatrixReduceOP::SUM, d_R, &d_R, Rcols, nblines, r_pitch, &r_pitch, nblines);
        std::cerr << "CROSS-COVS SUM DONE" << std::endl;

        // COV += COVS SUM
        matrix_op<double>(dim3(1, 1), dim3(covCols * covLines,1), d_cov, d_R, d_cov, MatrixOP::ADD,
             1, covCols * covLines, cov_pitch,
             1, Rcols, r_pitch,
             1, covCols * covLines, cov_pitch);
        std::cerr << "CROSS-COVS ADD DONE" << std::endl;
        Pstartindex += nblines;
    }
}
