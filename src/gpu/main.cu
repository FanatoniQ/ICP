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

__global__ void print_kernel()
{
    printf("Hello from block %d, thread %d\n", blockIdx.y * 10 + blockIdx.x, threadIdx.x);
}

// TODO: REMOVE ME since useless
__global__ void print_matrix_kernel(char *d_A, int pitch, int nbvals)
{
    int j;
    int idx = threadIdx.x;
    double *line = (double*)(d_A + idx * pitch);
    printf("Line %d:\n", idx);
    for (j = 0; j < nbvals; ++j) {
        //printf("%6.2f\t", (double)(d_A[idx * pitch + j * sizeof(double)]));
        printf("%6.2f\t", line[j]);
	__syncthreads();
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    std::string f1Header{};
    size_t Plines, Pcols;
    //___readCSV(f, f1Header);
    double *Pt = readCSV(argv[1], f1Header, Plines, Pcols);
    CPUMatrix P = CPUMatrix(Pt, Plines, Pcols);
    std::cout << P;

    double *Qt = malloc(sizeof(double) * Plines * Pcols);

    //std::vector<std::tuple<size_t, int>> correspondances = {};
    naiveGPUTranspose<<<32, 32>>>(Pt, Qt, Plines, Pcols);//(P, Q, correspondances);
    CPUMatrix Q = CPUMatrix(Qt, Pcols, Plines);
    std::cout << Q;
    cudaDeviceSynchronize();
    cudaCheckError();
}
