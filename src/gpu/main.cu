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
    float *line = (float*)(d_A + idx * pitch);
    printf("Line %d:\n", idx);
    for (j = 0; j < nbvals; ++j) {
        //printf("%6.2f\t", (float)(d_A[idx * pitch + j * sizeof(float)]));
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
        float min_dist = std::numeric_limits<float>::max();
        int chosen_idx = -1;
        for (size_t j = 0; j < Q.getDim0(); j++)
        {
            auto q_point = Q.getLine(j);
            float dist = std::sqrt(p_point.euclidianDistance(q_point));
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

float default_kernel(CPUMatrix a)
{
    UNUSED(a);
    return 1;
}

float default_kernel(float a)
{
    UNUSED(a);
    return 1;
}

// Implementation with CPUMAtrix
std::tuple<CPUMatrix, std::vector<float>> compute_cross_variance(CPUMatrix &P, CPUMatrix &Q,
                                                                  const std::vector<std::tuple<size_t, int>> &correspondences, float (*kernel)(CPUMatrix a))
{
    if (kernel == nullptr)
        kernel = &default_kernel;
    CPUMatrix cov = CPUMatrix(P.getDim1(), P.getDim1());
    std::vector<float> exclude_indices = {};
    for (auto tup : correspondences)
    {
        auto i = std::get<0>(tup);
        auto j = std::get<1>(tup);
        CPUView q_point = Q.getLine(j);
        CPUView p_point = P.getLine(i);
        float weight = kernel(p_point - q_point);

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
    runtime_assert(argc >= 4, "Usage: ./GPUICP file1 file2 nbiters");
    std::string f1Header{};
    size_t Qlines, Qcols, Plines, Pcols;
    //size_t Plines, Pcols;
    //___readCSV(f, f1Header);
    float *Pt = readCSV(argv[1], f1Header, Plines, Pcols);
    CPUMatrix P = CPUMatrix(Pt, Plines, Pcols);

    float *Qt = readCSV(argv[2], f1Header, Qlines, Qcols);
    CPUMatrix Q = CPUMatrix(Qt, Qlines, Qcols);

    unsigned int nbiters = std::stoi(argv[3]);
    CPUMatrix P_res;

    // FIXME iterations number
    if (argc == 5 && strcmp(argv[4], "-batch") == 0)
         P_res = icp_gpu(P, Q, nbiters);
    else
         P_res = icp_gpu_optimized(P, Q, nbiters);
    std::cout << "Squared actual mean diff: " << Q.euclidianDistance(P_res) << std::endl;
    //std::cout << "P resultat matrix: " << P_res;
    //std::cout << "Q ref matrix: " << Q;
    /*
    auto correspondances = get_correspondence_indices(P.getArray(), Q.getArray(), P.getDim0(), P.getDim1(), Q.getDim0(), Q.getDim1());
    for (int i = 0; i < 30; i++)
    {
        std::cout << std::get<0>(correspondances.at(i)) << " " << std::get<1>(correspondances.at(i)) << std::endl;
    }
    */
    //float *B = (float *)calloc(Plines*Pcols, sizeof(float));
    //float *B = calling_transpose_kernel(P.getArray(), Plines, Pcols);
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
    float A[9];
    float B[9];
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

    //float values = 0;
    //int row = Plines;
    //int column = Pcols;
    /*
    float *source, *dest;
    float *d_source, *d_dest;
    int row = 8;
    int column = 4;
    size_t size = row * column * sizeof(float);

    source = (float *)malloc(size);
    dest = (float *)malloc(size);

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
   

    float *B = calling_transpose_kernel(P.getArray(), Plines, Pcols);

    for (int i=0; i < column; ++i) {
        for (int j = 0; j < row; ++j) {
            std::cout<<B[i*row+j]<<' ';
        }
        std::cout<<std::endl;
    }
    //float *Qt = (float*)malloc(sizeof(float) * Plines * Pcols);
    
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
