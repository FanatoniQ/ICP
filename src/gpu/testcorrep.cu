#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <iostream>
#include <limits>

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

double randomdouble(double low, double high)
{
    return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    // simulating P matrix with 10 lines (points) and Q matrix with 5 lines (points) correspondences
    size_t dim0 = 10;
    size_t dim1 = 5;
    ICPCorresp *C = (ICPCorresp *)malloc(dim0 * dim1 * sizeof(ICPCorresp));
    for (size_t i = 0; i < dim0; ++i)
    {
        for (size_t j = 0; j < dim1; ++j)
        {
            C[i * dim1 + j] = {randomdouble(0.0, 10.0), (unsigned int)j};
            std::cerr << "dist: " << C[i * dim1 + j].dist << " id: " << C[i * dim1 + j].id << "\t";
        }
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
    for (size_t i = 0; i < dim0; ++i)
    {
        std::cerr << "dist: " << h_res[i * dim1].dist << " id: " << h_res[i * dim1].id << std::endl;
    }
    free(h_res);
    free(C);
    cudaFree(d_C);
    cudaCheckError();
}
