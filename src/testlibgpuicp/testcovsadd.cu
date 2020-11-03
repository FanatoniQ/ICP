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

__host__ float *get_cross_covs_cpu(CPUMatrix &P, size_t p_0, size_t p_1,
    CPUMatrix &Q, size_t q_0, size_t q_1,
    ICPCorresp *d_dist, size_t dist_0, size_t dist_1, size_t dist_pitch, size_t startindex)
{
    size_t ref_pitch = q_1 * p_1 * sizeof(float);
    float *h_ref = (float*)malloc(p_0 * ref_pitch);

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
        //std::cerr << cov.getDim0() * cov.getDim1() * sizeof(float) << std::endl;
        assert(ref_pitch == (cov.getDim0() * cov.getDim1() * sizeof(float)));
        memcpy(h_ref + i * ref_pitch / sizeof(float), cov.getArray(), cov.getDim0() * cov.getDim1() * sizeof(float));
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
    float *Pt = readCSV(argv[1], f1Header, Plines, Pcols);
    CPUMatrix P = CPUMatrix(Pt, Plines, Pcols);
    float *Qt = readCSV(argv[2], f1Header, Qlines, Qcols);
    CPUMatrix Q = CPUMatrix(Qt, Qlines, Qcols);

    // device P matrix
    size_t p_pitch = Pcols * sizeof(float);
    float *d_P;
    //cudaMallodist_pitch((void **)&d_P, &p_pitch, Pcols * sizeof(float), Plines);
    cudaMalloc((void**)&d_P, Plines * p_pitch);
    cudaCheckError();
    cudaMemcpy2D(d_P, p_pitch, Pt, Pcols * sizeof(float), Pcols * sizeof(float), Plines, cudaMemcpyHostToDevice);
    cudaCheckError();

    // device Q matrix
    size_t q_pitch = Qcols * sizeof(float);
    float *d_Q;
    //cudaMallodist_pitch((void **)&d_Q, &q_pitch, Qcols * sizeof(float), Qlines);
    cudaMalloc((void**)&d_Q, Qlines * q_pitch);
    cudaCheckError();
    cudaMemcpy2D(d_Q, q_pitch, Qt, Qcols * sizeof(float), Qcols * sizeof(float), Qlines, cudaMemcpyHostToDevice);
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
    size_t r_pitch = Rcols * sizeof(float);
    float *d_R;
    // or 2d...
    cudaMalloc((void**)&d_R, Rlines * r_pitch);
    cudaCheckError();

    // device FINAL cross-cov (flattened)
    size_t covLines = Qcols, covCols = Pcols;
    size_t cov_pitch = covCols * covLines * sizeof(float);
    float *d_cov;
    cudaMalloc((void**)&d_cov, 1 * cov_pitch);
    cudaMemset(d_cov, 0, cov_pitch);
    cudaCheckError();

    float ttlerror = 0;
    auto COV = CPUMatrix(Qcols, Pcols);
    auto RefCOV = CPUMatrix(Qcols, Pcols);

    while (Pstartindex < Plines)
    {
        nblines = MIN(Plines - Pstartindex, DISTS_LINES);

        // DISTS
        get_distances(d_P + Pstartindex * p_pitch / sizeof(float), d_Q, &d_dist, nblines, Pcols, p_pitch, Qlines, Qcols, q_pitch, nblines, dist_1, &dist_pitch, true);
        std::cerr << "DISTS DONE" << std::endl;

        // CORRESPS
        get_correspondences(d_dist, dist_pitch, nblines, dist_1, true);
        std::cerr << "CORRESPS DONE" << std::endl;

        /** Testing corresps: **/
        ICPCorresp *h_corresp = (ICPCorresp *)malloc(nblines * 1 * sizeof(ICPCorresp));
        cudaMemcpy2D(h_corresp, 1 * sizeof(ICPCorresp), d_dist, dist_pitch, 1 * sizeof(ICPCorresp), nblines, cudaMemcpyDeviceToHost);
        cudaCheckError();
	//if (Pstartindex >= Plines - 100)
	if (Pstartindex < 100)
{	for (size_t i = 0; i < nblines; ++i)
	{
	//if (i + Pstartindex >= Plines - 100)
            std::cout << i + Pstartindex << " -> " << h_corresp[i].id << " , " << h_corresp[i].dist << std::endl;
	}
}
	free(h_corresp);


        // CROSS-COVS
        get_cross_cov(d_P + Pstartindex * p_pitch / sizeof(float), d_Q, &d_R, d_dist,
            nblines, Pcols, p_pitch,
            Qlines, Qcols, q_pitch,
            nblines, Rcols, &r_pitch,
            nblines, dist_1, dist_pitch, true);
        std::cerr << "CROSS-COVS DONE" << std::endl;

        /** Testing cross-covs: **/
        float *h_ref_cross_covs = get_cross_covs_cpu(P, nblines, Pcols, Q, Qlines, Qcols, d_dist, nblines, dist_1, dist_pitch, Pstartindex);
        float *h_r = (float*)malloc(nblines * Rcols * sizeof(float));
        cudaMemcpy2D(h_r, Rcols * sizeof(float), d_R, r_pitch, Rcols * sizeof(float), nblines, cudaMemcpyDeviceToHost);
        cudaCheckError();
        //auto BatchRefCOV = CPUMatrix(h_ref_cross_covs, Qcols, Pcols);
        auto BatchRefCOV = CPUMatrix(Qcols, Pcols);
        //assert(memcmp(h_ref_cross_covs, h_r, Rlines * Rcols * sizeof(float)) == 0);
	ttlerror = 0;
        for (size_t i = 0; i < nblines; i++)
        {
            for (size_t j = 0; j < Rcols; ++j)
            {
                float error = std::fabs(h_r[i * (r_pitch / sizeof(float)) + j] - h_ref_cross_covs[i * (r_pitch / sizeof(float)) + j]); // Weird not having to divide by sizeof float...
                std::cerr << h_r[i * (r_pitch / sizeof(float)) + j] << " \t " <<  h_ref_cross_covs[i * (r_pitch / sizeof(float)) + j] << std::endl;
                ttlerror += error;
            }
	    auto LineRefCOV = CPUMatrix(h_ref_cross_covs + i * (r_pitch / sizeof(float)), Qcols, Pcols);
	    BatchRefCOV += LineRefCOV;
	    LineRefCOV.setArray(nullptr,1,1);
        }
        std::cerr << "Error (batch cross-covs): " << ttlerror << std::endl;
        std::cerr << "Mean Error (batch cross-covs): " << ttlerror / nblines * Rcols << std::endl;
	//assert(Qcols * Pcols == nblines * Rcols); // should break
        RefCOV += BatchRefCOV;
        free(h_r);

        // COVS SUM
	//cudaMemset(d_R, 0, r_pitch * Rlines);
        //reduce_0(MatrixReduceOP::SUM, d_dist, float **d_sum, Pcols * Qcols, Plines, dist_pitch, size_t *reducepitch, int threads);
        reduce_0(MatrixReduceOP::SUM, d_R, &d_R, Rcols, nblines, r_pitch, &r_pitch, nblines);

        // COV += COVS SUM
        assert(covCols == 3 && covLines == 3);
        matrix_op<float>(dim3(1, 1), dim3(covCols * covLines,1), d_cov, d_R, d_cov, MatrixOP::ADD,
             1, covCols * covLines, cov_pitch,
             1, Rcols, r_pitch,
             1, covCols * covLines, cov_pitch);

        /** testing covs-sum **/
        /**for (size_t i = 0; i < Rlines; i++)
        {
            auto c = CPUMatrix(h_ref_cross_covs + i * (r_pitch / sizeof(float)), Qcols, Pcols);
        RefCOV += c;
        c.setArray(nullptr,1,1); // avoid freeing
        }**/
        // TODO: do this on GPU
        float *h_cov = (float *)malloc(Rcols * sizeof(float));
        cudaMemcpy(h_cov, d_R, Rcols * sizeof(float), cudaMemcpyDeviceToHost);
        auto BatchCOV = CPUMatrix(h_cov, Qcols, Pcols);
        COV += BatchCOV;

	ttlerror = 0;
        for (size_t i = 0; i < Qcols; ++i)
        {
            for (size_t j = 0; j < Pcols; ++j)
            {
                float error = std::fabs(BatchCOV(i,j) - BatchRefCOV(i,j));
				//h_cov[i * Pcols + j] - h_ref_cross_covs[i * (r_pitch / sizeof(float)) + j]); // Weird not having to divide by sizeof float...
                //std::cerr << h_r[i * (r_pitch / sizeof(float)) + j] << " \t " <<  h_ref_cross_covs[i * (r_pitch / sizeof(float)) + j] << std::endl;
                ttlerror += error;
            }
        }
	std::cout << "Error (batch-reduced cross-cov): " << ttlerror << std::endl;
        std::cout << "Mean Error (batch-reduced cross-cov): " << ttlerror / (Pcols * Qcols) << std::endl;

        std::cerr << "BatchRefCOV:" << std::endl;
        std::cerr << BatchRefCOV << std::endl;

        std::cerr << "BatchCOV:" << std::endl;
        std::cerr << BatchCOV << std::endl;

	//assert(BatchRefCOV.getArray() == BatchCOV.getArray());
	std::cout << "index: " << Pstartindex << "/" << Plines << std::endl;

        Pstartindex += nblines;
    }
    //auto COV = CPUMatrix(h_cov, Qcols, Pcols);
    ttlerror = 0;
    for (size_t i = 0; i < Qcols; ++i)
    {
        for (size_t j = 0; j < Pcols; ++j)
        {
            float error = std::fabs(RefCOV(i,j) - COV(i,j));
            ttlerror += error;
        }
    }
    std::cout << "Error (FINAL CPU summed cross-cov): " << ttlerror << std::endl;
    std::cout << "Mean Error (FINAL CPU summed cross-cov): " << ttlerror / (Pcols * Qcols) << std::endl;

    std::cout << "CPURefCOV:" << std::endl;
    std::cout << RefCOV << std::endl;

    std::cout << "CPUsummed GPUCOVs:" << std::endl;
    std::cout << COV << std::endl;

    float *h_cov = (float *)malloc(covLines * covCols * sizeof(float));
    cudaMemcpy(h_cov, d_cov, covLines * covCols * sizeof(float), cudaMemcpyDeviceToHost);
    auto FULLGPUCOV = CPUMatrix(h_cov, covLines, covCols);
    
    ttlerror = 0;
    for (size_t i = 0; i < Qcols; ++i)
    {
        for (size_t j = 0; j < Pcols; ++j)
        {
            float error = std::fabs(RefCOV(i,j) - FULLGPUCOV(i,j));
            ttlerror += error;
        }
    }
    std::cout << "Error (FINAL FULLGPU cross-cov): " << ttlerror << std::endl;
    std::cout << "Mean Error (FINAL FULLGPU cross-cov): " << ttlerror / (Pcols * Qcols) << std::endl;

    std::cout << "FULL GPUCOV:" << std::endl;
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
