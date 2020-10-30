#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <iostream>

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
    // simulating P matrix with 10 lines (points) and Q matrix with 5 lines (points) correspondences
    size_t dim0 = 10;
    size_t dim1 = 5;
    struct Corresp *C = (struct Corresp *)malloc(dim0 * dim1 * sizeof(struct Corresp));
    for (size_t i = 0; i < dim0; ++i)
    {
        for (size_t j = 0; j < dim1; ++j)
        {
            C[i * dim1 + dim0] = {randomdouble(0.0, 10.0), j};
            std::cerr << "dist: " << C[i * dim1 + dim0].dist << " id: " << C[i * dim1 + dim0].id << std::endl;
        }
    }

    size_t reducepitch;
    struct Corresp *d_C;
    cudaMallocPitch(&d_C, &reducepitch, dim1 * sizeof(struct Corresp), dim0);
    cudaCheckError();
    cudaMemcpy2D(d_C, reducepitch, C, dim1 * sizeof(struct Corresp), dim1 * sizeof(struct Corresp), dim0, cudaMemcpyHostToDevice);
    cudaCheckError();

    get_correspondences(d_C, reducepitch, dim0, dim1, true);

    double *h_res = (struct Corresp *)malloc(dim0 * dim1 * sizeof(struct Corresp));
    cudaMemcpy2D(h_res, dim1 * sizeof(struct Corresp), d_C, reducepitch, 1 * sizeof(struct Corresp), dim0, cudaMemcpyDeviceToHost);
    cudaCheckError();

    std::cerr << "Min dist: " << std::endl;
    for (size_t i = 0; i < dim0; ++i)
    {
        for (size_t j = 0; j < dim1; ++j)
        {
            std::cerr << "dist: " << h_res[i * dim1 + dim0].dist << " id: " << h_res[i * dim1 + dim0].id << std::endl;
        }
    }
    free(h_res);
    free(C);
    cudaFree(d_C);
    cudaCheckError();
}