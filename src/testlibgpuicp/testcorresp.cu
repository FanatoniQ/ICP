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
//#include "gpu/icp.cuh"
#include "libgpuicp/corresp.cuh"

double randomdouble(double low, double high)
{
    return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    // simulating P matrix with 10 lines (points) and Q matrix with 5 lines (points) correspondences
    size_t dim0 = 100;
    size_t dim1 = 3;
    ICPCorresp *C = (ICPCorresp *)malloc(dim0 * dim1 * sizeof(ICPCorresp));
    ICPCorresp *Cmin = (ICPCorresp *)malloc(dim0 * dim1 * sizeof(ICPCorresp));
    for (size_t i = 0; i < dim0; ++i)
    {
	size_t minid;
	double min = DBL_MAX;
        for (size_t j = 0; j < dim1; ++j)
        {
            C[i * dim1 + j] = {randomdouble(0.0, 10.0), (unsigned int)j};
	    if (C[i * dim1 + j].dist < min) {
                 min = C[i * dim1 + j].dist;
		 minid = j;
            }
            std::cerr << "dist: " << C[i * dim1 + j].dist << " id: " << C[i * dim1 + j].id << "\t";
        }
	Cmin[i * dim1] = {min, (unsigned int)minid};
	std::cerr << std::endl;
    }

    size_t reducepitch;
    ICPCorresp *d_C;
    cudaMallocPitch((void **)&d_C, &reducepitch, dim1 * sizeof(ICPCorresp), dim0); // FIXME: crashes...
    cudaCheckError();
    cudaMemcpy2D(d_C, reducepitch, C, dim1 * sizeof(ICPCorresp), dim1 * sizeof(ICPCorresp), dim0, cudaMemcpyHostToDevice);
    cudaCheckError();

    get_correspondences(d_C, reducepitch, dim0, dim1, true);
    std::cerr << "DONE" << std::endl;

    ICPCorresp *h_res = (ICPCorresp *)malloc(dim0 * dim1 * sizeof(ICPCorresp));
    cudaMemcpy2D(h_res, dim1 * sizeof(ICPCorresp), d_C, reducepitch, 1 * sizeof(ICPCorresp), dim0, cudaMemcpyDeviceToHost);
    cudaCheckError();

    std::cerr << "Min dist: " << std::endl;
    size_t errors = 0;
    for (size_t i = 0; i < dim0; ++i)
    {
        std::cerr << "dist: " << h_res[i * dim1].dist << " id: " << h_res[i * dim1].id << std::endl;
	if (h_res[i * dim1].id != Cmin[i * dim1].id)
	{
             errors++;
             std::cerr << "DIFFERENCE ! " << std::endl << "refdist: " << Cmin[i * dim1].dist << " refid: " << Cmin[i * dim1].id << std::endl;
	}
    }
    std::cerr << "ERRORS:" << errors << std::endl;
    free(h_res);
    free(C);
    cudaFree(d_C);
    cudaCheckError();


    std::string f1Header{};
    size_t Qlines, Qcols, Plines, Pcols;
    //size_t Plines, Pcols;
    //___readCSV(f, f1Header);
    double* Pt = readCSV(argv[1], f1Header, Plines, Pcols);
    for (int i = 0; i < 30; i++)
        std::cout << Pt[i] << std::endl;
    double* Qt = readCSV(argv[2], f1Header, Qlines, Qcols);

    double* d_P, * d_Q;

    cudaMalloc(&d_P, sizeof(double) * Plines * Pcols);
    cudaMalloc(&d_Q, sizeof(double) * Qlines * Qcols);

    cudaMemcpy(d_P, Pt, sizeof(double) * Pcols * Plines, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, Qt, sizeof(double) * Qcols * Qlines, cudaMemcpyHostToDevice);

    unsigned int* d_array_correspondances;
    cudaMalloc(&d_array_correspondances, sizeof(unsigned int) * Plines);

    get_array_correspondences(d_array_correspondances, d_P, d_Q, Plines, Pcols, Qlines, Qcols);

    cudaMemcpy(Qt, d_Q, sizeof(double) * Qcols, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 30; i++)
        std::cout << Qt[i] << std::endl;

    cudaFree(d_P);
    cudaFree(d_Q);
    cudaFree(d_array_correspondances);
    free(Pt);
    free(Qt);
}
