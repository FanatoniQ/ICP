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
#include "libgpualg/mean.cuh"
#include "error.cuh"
//#include "gpu/icp.cuh"
#include "libgpuicp/dist.cuh"
#include "libgpuicp/corresp.cuh"

int main(int argc, char **argv)
{
    std::string f1Header{};
    size_t Qlines, Qcols, Plines, Pcols;
    double *Pt = readCSV(argv[1], f1Header, Plines, Pcols);
    CPUMatrix P = CPUMatrix(Pt, Plines, Pcols);
    double *Qt = readCSV(argv[2], f1Header, Qlines, Qcols);
    CPUMatrix Q = CPUMatrix(Qt, Qlines, Qcols);

    ICPCorresp *h_ref_dist = (ICPCorresp *)malloc(Plines * Qlines * sizeof(ICPCorresp));
    runtime_assert(h_ref_dist != nullptr, "Invalid ptr");
    for (size_t i = 0; i < Plines; ++i)
    {
         for (size_t j = 0; j < Qlines; ++j)
         {
             auto dist = P.getLine(i).euclidianDistance(Q.getLine(j));
	     assert(P.getLine(i).euclidianDistance(Q.getLine(j)) == Q.getLine(j).euclidianDistance(P.getLine(i)));
             h_ref_dist[i * Qlines + j] = {dist, (unsigned int)j};
         }
    }

    // device P matrix
    size_t p_pitch = Pcols * sizeof(double);
    double *d_P;
    //cudaMallocPitch((void **)&d_P, &p_pitch, Pcols * sizeof(double), Plines);
    cudaMalloc((void**)&d_P, Plines * p_pitch);
    cudaCheckError();
    cudaMemcpy2D(d_P, p_pitch, Pt, Pcols * sizeof(double), Pcols * sizeof(double), Plines, cudaMemcpyHostToDevice);
    cudaCheckError();

    // device Q matrix
    size_t q_pitch = Qcols * sizeof(double);
    double *d_Q;
    //cudaMallocPitch((void **)&d_Q, &q_pitch, Qcols * sizeof(double), Qlines);
    cudaMalloc((void**)&d_Q, Qlines * q_pitch);
    cudaCheckError();
    cudaMemcpy2D(d_Q, q_pitch, Qt, Qcols * sizeof(double), Qcols * sizeof(double), Qlines, cudaMemcpyHostToDevice);
    cudaCheckError();

    // device dist matrix
    size_t dist_pitch;
    ICPCorresp *d_dist;
    cudaMallocPitch((void **)&d_dist, &dist_pitch, Qlines * sizeof(ICPCorresp), Plines);
    cudaCheckError();

    // call kernel
    get_distances(d_P, d_Q, &d_dist, Plines, Pcols, p_pitch, Qlines, Qcols, q_pitch, Plines, Qlines, &dist_pitch, true);
    std::cerr << "DONE" << std::endl;

    // copy back to host
    // host dist matrix
    ICPCorresp *h_dist = (ICPCorresp *)malloc(Plines * Qlines * sizeof(ICPCorresp));
    cudaMemcpy2D(h_dist, Qlines * sizeof(ICPCorresp), d_dist, dist_pitch, Qlines * sizeof(ICPCorresp), Plines, cudaMemcpyDeviceToHost);
    cudaCheckError();

    double ttlerror = 0;
    for (size_t i = 0; i < Plines; ++i)
    {
         for (size_t j = 0; j < Qlines; ++j)
         {
             //std::cerr << "dist: " << h_dist[i * Qlines + j].dist << " id: " <<  h_dist[i * Qlines + j].id << "\t";
	     if (h_dist[i * Qlines + j].id != h_ref_dist[i * Qlines + j].id)
	     {
		     std::cerr << "FATAL ID ERROR !" << std::endl;
		     return EXIT_FAILURE;
	     }
	     double err = std::fabs(h_dist[i * Qlines + j].dist - h_ref_dist[i * Qlines + j].dist);
	     ttlerror += err;
             std::cerr << err << "\t";
             //if (memcmp(&h_dist[i * Qlines + j], &h_ref_dist[i * Qlines + j], sizeof(ICPCorresp)) != 0)
             //{
                 //std::cerr << "h_ref_dist: " << h_ref_dist[i * Qlines + j].dist << " id: " << h_ref_dist[i * Qlines + j].id << std::endl;
		 //return EXIT_FAILURE;
             //}
         }
	 std::cerr << std::endl;
    }
    std::cerr << std::endl << "ttlerror: " << ttlerror << std::endl;
    std::cerr << "mean error: " << ttlerror / (Plines * Qlines) << std::endl;
    std::cerr << "SUCCESS" << std::endl;

    free(h_dist);
    free(h_ref_dist);
    cudaFree(d_P);
    cudaCheckError();
    cudaFree(d_Q);
    cudaCheckError();
    cudaFree(d_dist);
    cudaCheckError();
}
