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

// NEED TO DELETE THIS
std::vector<std::tuple<size_t, int>> get_correspondence_indices(CPUMatrix &P, CPUMatrix &Q)
{
    std::vector<std::tuple<size_t, int>> correspondances = {};
    for (size_t i = 0; i < P.getDim0(); i++)
    {
        auto p_point = P.getLine(i);
        double min_dist = std::numeric_limits<double>::max();
        int chosen_idx = -1;
        for (size_t j = 0; j < Q.getDim0(); j++)
        {
            auto q_point = Q.getLine(j);
            double dist = std::sqrt(p_point.euclidianDistance(q_point));
            if (dist < min_dist)
            {
                min_dist = dist;
                chosen_idx = j;
            }
        }
        correspondances.push_back(std::make_tuple(i, chosen_idx));
    }
    return correspondances;
}

double default_kernel(CPUMatrix a)
{
    UNUSED(a);
    return 1;
}

double default_kernel(double a)
{
    UNUSED(a);
    return 1;
}

// Implementation with CPUMAtrix
std::tuple<CPUMatrix, std::vector<double>> compute_cross_variance(CPUMatrix &P, CPUMatrix &Q,
                                                                  const std::vector<std::tuple<size_t, int>> &correspondences, double (*kernel)(CPUMatrix a))
{
    if (kernel == nullptr)
        kernel = &default_kernel;
    CPUMatrix cov = CPUMatrix(P.getDim1(), P.getDim1());
    std::vector<double> exclude_indices = {};
    for (auto tup : correspondences)
    {
        auto i = std::get<0>(tup);
        auto j = std::get<1>(tup);
        CPUView q_point = Q.getLine(j);
        CPUView p_point = P.getLine(i);
        double weight = kernel(p_point - q_point);

        if (weight < 0.01)
            exclude_indices.push_back(i);

        CPUMatrix doted_points = q_point.transpose().dot(p_point);
        doted_points *= weight;
        cov += doted_points;
    }
    return std::make_tuple(std::move(cov), exclude_indices);
}

int main(int argc, char **argv)
{
    std::string f1Header{};
    size_t Qlines, Qcols, Plines, Pcols;
    //size_t Plines, Pcols;
    //___readCSV(f, f1Header);
    double *Pt = readCSV(argv[1], f1Header, Plines, Pcols);
    CPUMatrix P = CPUMatrix(Pt, Plines, Pcols);

    double *Qt = readCSV(argv[2], f1Header, Qlines, Qcols);
    CPUMatrix Q = CPUMatrix(Qt, Qlines, Qcols);

    // FIXME iterations number
    auto P_res = icp_gpu(P, Q, 10);
    std::cout << "Squared actual mean diff: " << Q.euclidianDistance(P_res) << std::endl;
    std::cout << "P resultat matrix: " << P_res;
    std::cout << "Q ref matrix: " << Q;
    /*
    auto correspondances = get_correspondence_indices(P.getArray(), Q.getArray(), P.getDim0(), P.getDim1(), Q.getDim0(), Q.getDim1());
    for (int i = 0; i < 30; i++)
    {
        std::cout << std::get<0>(correspondances.at(i)) << " " << std::get<1>(correspondances.at(i)) << std::endl;
    }
    */
    //double *B = (double *)calloc(Plines*Pcols, sizeof(double));
    //double *B = calling_transpose_kernel(P.getArray(), Plines, Pcols);
    //for (int i = 0; i < Plines; i++)
    //{
    //    for (int j = 0; j < Pcols; j++)
    //    {
    //        std::cout << B[i*Pcols + j] << " " << Pt[j*Pcols + i];
    //    }
    //    std::cout << std::endl;
    //}

    //auto correspondences = get_correspondence_indices(P, Q);
    //auto finale = compute_cross_variance(P, Q, correspondences, nullptr);
    //auto cov = compute_cross_variance_cpu_call_gpu(P.getArray(), Q.getArray(), correspondences, P.getDim0(), P.getDim1(), Q.getDim0(), Q.getDim1());

    //std::cout << std::get<0>(finale) << std::endl;
    /*
    double A[9];
    double B[9];
    for (int i = 0; i < 9; i++)
    {
        A[i] = 1;
        B[i] = 1;
    }
    auto cov = calling_dot_kernel(A, B, 3, 3, 3, 3);
    for (int i = 0; i < 9; i++)
        std::cout << *(cov + i) << std::endl;
    
    free(cov);
    */

    //double values = 0;
    //int row = Plines;
    //int column = Pcols;
    /*
    double *source, *dest;
    double *d_source, *d_dest;
    int row = 8;
    int column = 4;
    size_t size = row * column * sizeof(double);

    source = (double *)malloc(size);
    dest = (double *)malloc(size);

    cudaMalloc((void **)&d_source, size);
    cudaMalloc((void **)&d_dest, size);
    
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            Pt[i*column+j] = values;
            values++;
        }
    }

    
    cudaMemcpy(d_source, source, size, cudaMemcpyHostToDevice);

    gpuTranspose(d_source, d_dest, row, column);

    cudaMemcpy(dest, d_dest, size, cudaMemcpyDeviceToHost);
   

    double *B = calling_transpose_kernel(P.getArray(), Plines, Pcols);

    for (int i=0; i < column; ++i) {
        for (int j = 0; j < row; ++j) {
            std::cout<<B[i*row+j]<<' ';
        }
        std::cout<<std::endl;
    }
    //double *Qt = (double*)malloc(sizeof(double) * Plines * Pcols);
    
    free(source);
    free(dest);
    cudaFree(d_source);
    cudaFree(d_dest);
    //std::vector<std::tuple<size_t, int>> correspondances = {};
    //naiveGPUTranspose<<<32, 32>>>(Pt, Qt, Plines, Pcols);//(P, Q, correspondances);
    //CPUMatrix Q = CPUMatrix(Qt, Pcols, Plines);
    //std::cout << Q;
    cudaDeviceSynchronize();
    cudaCheckError();
    */
}