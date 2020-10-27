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


    double values = 0;
    double *source, *dest;
    double *d_source, *d_dest;
    int row = 30;
    int column = 3;
    size_t size = row * column * sizeof(double);

    source = (double *)malloc(size);
    dest = (double *)malloc(size);

    cudaMalloc((void **)&d_source, size);
    cudaMalloc((void **)&d_dest, size);

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            source[i*column+j] = values;
            values++;
        }
    }

    cudaMemcpy(d_source, source, size, cudaMemcpyHostToDevice);
    naiveGPUTranspose<<<32, 32>>>(d_source, d_dest, row, column);
    cudaMemcpy(dest, d_dest, size, cudaMemcpyDeviceToHost);
    
    for (int i=0; i < column; ++i) {
        for (int j = 0; j < row; ++j) {
            std::cout<<dest[i*row+j]<<' ';
        }
        std::cout<<std::endl;
    }
    //double *Qt = (double*)malloc(sizeof(double) * Plines * Pcols);

    //std::vector<std::tuple<size_t, int>> correspondances = {};
    //naiveGPUTranspose<<<32, 32>>>(Pt, Qt, Plines, Pcols);//(P, Q, correspondances);
    //CPUMatrix Q = CPUMatrix(Qt, Pcols, Plines);
    //std::cout << Q;
    cudaDeviceSynchronize();
    cudaCheckError();
}
