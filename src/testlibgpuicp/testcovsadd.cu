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
//#include "gpu/icp.cuh"
#include "libgpuicp/corresp.cuh"
#include "libgpuicp/dist.cuh"
#include "libgpuicp/corresp.cuh"
#include "libgpuicp/crosscov.cuh"

#define DISTS_LINES 128

__host__ double *get_cross_covs_cpu(CPUMatrix &P, size_t p_0, size_t p_1,
    CPUMatrix &Q, size_t q_0, size_t q_1,
    ICPCorresp *d_dist, size_t dist_0, size_t dist_1, size_t dist_pitch, size_t startindex)
{
    size_t ref_pitch = q_1 * p_1 * sizeof(double);
    double *h_ref = (double*)malloc(p_0 * ref_pitch);

    size_t h_dist_pitch = dist_1 * sizeof(ICPCorresp);
    ICPCorresp *h_dist = (ICPCorresp*)malloc(dist_0 * h_dist_pitch);

    cudaMemcpy2D(h_dist, h_dist_pitch, d_dist, dist_pitch, dist_1 * sizeof(ICPCorresp), dist_0, cudaMemcpyDeviceToHost);
    cudaCheckError();

    for (size_t i = 0; i < p_0; ++i)
    {
        size_t idp = i + startindex;
        size_t idq = h_dist[i * dist_1].id;
        std::cerr << "idq: " << idq << "idp: " << idp << std::endl;
        auto cov = Q.getLine(idq).transpose().dot(P.getLine(idp)); // since getLine returns line vector
        //std::cerr << ref_pitch << std::endl;
        //std::cerr << cov.getDim0() * cov.getDim1() * sizeof(double) << std::endl;
        assert(ref_pitch == (cov.getDim0() * cov.getDim1() * sizeof(double)));
        memcpy(h_ref + i * ref_pitch / sizeof(double), cov.getArray(), cov.getDim0() * cov.getDim1() * sizeof(double));
        for (size_t a = 0; a < q_1 * p_1; ++a)
            std::cerr << cov.getArray()[a] << "\t";
        std::cerr << std::endl;
    }

    free(h_dist);

    return h_ref;
}

int main(int argc, char **argv)
{
    runtime_assert(argc == 3, "./bin file1 file2");
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

    //size_t nbiters =  std::ceil((float)Plines / DISTS_LINES);
    size_t Pstartindex = 0;
    size_t nblines;

    // device dist,id distance and corresp matrix
    size_t dist_0 = DISTS_LINES, dist_1 = Qlines;
    size_t dist_pitch;
    ICPCorresp *d_dist;
    cudaMallocPitch((void **)&d_dist, &dist_pitch, dist_1 * sizeof(ICPCorresp), dist_0);
    cudaCheckError();
    //cudaMemcpy2D(d_dist, dist_pitch, C, Qlines * sizeof(ICPCorresp), Qlines * sizeof(ICPCorresp), Plines, cudaMemcpyHostToDevice);
    //cudaCheckError();

    // device cross-covs flattened
    size_t Rlines = DISTS_LINES, Rcols = Pcols * Qcols;
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
    cudaMemset(d_cov, 0, cov_pitch);
    cudaCheckError();

    double ttlerror = 0;
    auto COV = CPUMatrix(Qcols, Pcols);
    auto RefCOV = CPUMatrix(Qcols, Pcols);

    while (Pstartindex < Plines)
    {
        nblines = MIN(Plines - Pstartindex, DISTS_LINES);

        // DISTS
        get_distances(d_P + Pstartindex * p_pitch / sizeof(double), d_Q, &d_dist, nblines, Pcols, p_pitch, Qlines, Qcols, q_pitch, nblines, dist_1, &dist_pitch, true);
        std::cerr << "DISTS DONE" << std::endl;

        // CORRESPS
        get_correspondences(d_dist, dist_pitch, nblines, dist_1, true);
        std::cerr << "CORRESPS DONE" << std::endl;

        /** Testing corresps: **/
        ICPCorresp *h_corresp = (ICPCorresp *)malloc(nblines * 1 * sizeof(ICPCorresp));
        cudaMemcpy2D(h_corresp, 1 * sizeof(ICPCorresp), d_dist, dist_pitch, 1 * sizeof(ICPCorresp), nblines, cudaMemcpyDeviceToHost);
        cudaCheckError();
	    if (Pstartindex < 100)
        {	
            for (size_t i = 0; i < nblines; ++i)
	        {
                std::cerr << i + Pstartindex << " -> " << h_corresp[i].id << " , " << h_corresp[i].dist << std::endl;
	        }
        }
	    free(h_corresp);

        // CROSS-COVS
        get_cross_cov(d_P + Pstartindex * p_pitch / sizeof(double), d_Q, &d_R, d_dist,
            nblines, Pcols, p_pitch,
            Qlines, Qcols, q_pitch,
            nblines, Rcols, &r_pitch,
            nblines, dist_1, dist_pitch, true);
        std::cerr << "CROSS-COVS DONE" << std::endl;

        /** Testing cross-covs: **/
        double *h_ref_cross_covs = get_cross_covs_cpu(P, nblines, Pcols, Q, Qlines, Qcols, d_dist, nblines, dist_1, dist_pitch, Pstartindex);
        double *h_r = (double*)malloc(nblines * Rcols * sizeof(double));
        cudaMemcpy2D(h_r, Rcols * sizeof(double), d_R, r_pitch, Rcols * sizeof(double), nblines, cudaMemcpyDeviceToHost);
        cudaCheckError();
        //auto BatchRefCOV = CPUMatrix(h_ref_cross_covs, Qcols, Pcols);
        auto BatchRefCOV = CPUMatrix(Qcols, Pcols);
        //assert(memcmp(h_ref_cross_covs, h_r, Rlines * Rcols * sizeof(double)) == 0);
	    ttlerror = 0;
        for (size_t i = 0; i < nblines; i++)
        {
            for (size_t j = 0; j < Rcols; ++j)
            {
                double error = std::fabs(h_r[i * (r_pitch / sizeof(double)) + j] - h_ref_cross_covs[i * (r_pitch / sizeof(double)) + j]); // Weird not having to divide by sizeof double...
                std::cerr << h_r[i * (r_pitch / sizeof(double)) + j] << " \t " <<  h_ref_cross_covs[i * (r_pitch / sizeof(double)) + j] << std::endl;
                ttlerror += error;
            }
	        auto LineRefCOV = CPUMatrix(h_ref_cross_covs + i * (r_pitch / sizeof(double)), Qcols, Pcols);
	        BatchRefCOV += LineRefCOV;
	        LineRefCOV.setArray(nullptr,1,1);
        }
        std::cerr << "Error (batch cross-covs): " << ttlerror << std::endl;
        std::cerr << "Mean Error (batch cross-covs): " << ttlerror / nblines * Rcols << std::endl;
        RefCOV += BatchRefCOV;
        free(h_r);

        // COVS SUM
        //reduce_0(MatrixReduceOP::SUM, d_dist, double **d_sum, Pcols * Qcols, Plines, dist_pitch, size_t *reducepitch, int threads);
        reduce_0(MatrixReduceOP::SUM, d_R, &d_R, Rcols, nblines, r_pitch, &r_pitch, nblines);

        // COV += COVS SUM
        assert(covCols == 3 && covLines == 3);
        matrix_op<double>(dim3(1, 1), dim3(covCols * covLines,1), d_cov, d_R, d_cov, MatrixOP::ADD,
             1, covCols * covLines, cov_pitch,
             1, Rcols, r_pitch,
             1, covCols * covLines, cov_pitch);

        /** testing covs-sum **/
        // TODO: do this on GPU
        double *h_cov = (double *)malloc(Rcols * sizeof(double));
        cudaMemcpy(h_cov, d_R, Rcols * sizeof(double), cudaMemcpyDeviceToHost);
        auto BatchCOV = CPUMatrix(h_cov, Qcols, Pcols);
        COV += BatchCOV;

	ttlerror = 0;
        for (size_t i = 0; i < Qcols; ++i)
        {
            for (size_t j = 0; j < Pcols; ++j)
            {
                double error = std::fabs(BatchCOV(i,j) - BatchRefCOV(i,j));
                ttlerror += error;
            }
        }
	std::cerr << "Error (batch-reduced cross-cov): " << ttlerror << std::endl;
        std::cerr << "Mean Error (batch-reduced cross-cov): " << ttlerror / (Pcols * Qcols) << std::endl;

        std::cerr << "BatchRefCOV:" << std::endl;
        std::cerr << BatchRefCOV << std::endl;

        std::cerr << "BatchCOV:" << std::endl;
        std::cerr << BatchCOV << std::endl;

	    std::cerr << "index: " << Pstartindex << "/" << Plines << std::endl;

        Pstartindex += nblines;
    }
    ttlerror = 0;
    for (size_t i = 0; i < Qcols; ++i)
    {
        for (size_t j = 0; j < Pcols; ++j)
        {
            double error = std::fabs(RefCOV(i,j) - COV(i,j));
            ttlerror += error;
        }
    }
    std::cerr << "Error (FINAL CPU summed cross-cov): " << ttlerror << std::endl;
    std::cerr << "Mean Error (FINAL CPU summed cross-cov): " << ttlerror / (Pcols * Qcols) << std::endl;

    std::cerr << "CPURefCOV:" << std::endl;
    std::cerr << RefCOV << std::endl;

    std::cerr << "CPUsummed GPUCOVs:" << std::endl;
    std::cerr << COV << std::endl;

    double *h_cov = (double *)malloc(covLines * covCols * sizeof(double));
    cudaMemcpy(h_cov, d_cov, covLines * covCols * sizeof(double), cudaMemcpyDeviceToHost);
    auto FULLGPUCOV = CPUMatrix(h_cov, covLines, covCols);
    
    ttlerror = 0;
    for (size_t i = 0; i < Qcols; ++i)
    {
        for (size_t j = 0; j < Pcols; ++j)
        {
            double error = std::fabs(RefCOV(i,j) - FULLGPUCOV(i,j));
            ttlerror += error;
        }
    }
    std::cerr << "Error (FINAL FULLGPU cross-cov): " << ttlerror << std::endl;
    std::cerr << "Mean Error (FINAL FULLGPU cross-cov): " << ttlerror / (Pcols * Qcols) << std::endl;

    std::cerr << "FULL GPUCOV:" << std::endl;
    std::cerr << FULLGPUCOV << std::endl;
    
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
