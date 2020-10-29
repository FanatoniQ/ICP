#include <vector>
#include <limits>
#include <tuple>
#include <iostream>
#include <cmath>

#include "libalg/basic_operations.hpp"
#include "libalg/alg.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/CPUView.hpp"
#include "error.hpp"
#include "cpu/tuple.hpp"

#include "gpu/icp.cuh"
#include "libgpualg/mult.cuh"

#define Tile_size 2

/* --------- CPU Version Calling GPU Kernel ------------ */
__host__ std::vector<std::tuple<size_t, int>> get_correspondence_indices(double *P, double *Q,
                                                                size_t P_r, size_t P_c, size_t Q_r, size_t Q_c)
{
    std::vector<std::tuple<size_t, int>> correspondances = {};
    for (size_t i = 0; i < P_r; i++)
    {
        double *p_point = P + i * P_c;
        double min_dist = std::numeric_limits<double>::max();
        int chosen_idx = -1;
        for (size_t j = 0; j < Q_r; j++)
        {
            double *q_point = Q + j * Q_c;
            double dist = std::sqrt(element_wise_reduce(p_point, q_point, 1, P_c, 1, Q_c,
                                    squared_norm_2, add, add)); //norm 2 between 2 vectors
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


// Intermediation function to be replaced with element_wise_op
__host__ void increment_cov(double *P, double *Q)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            P[i*3 + j] = P[i*3 + j] + Q[i*3 + j];
        }
    }
}

__host__ double* calling_transpose_kernel(double *A, size_t row, size_t column)
{
        // Calling transpose kernel
        size_t size = sizeof(double) * row * column;

        // Allocations
        double *d_source_transpose, *d_dest_transpose;
        cudaMalloc((void **)&d_source_transpose, size);
        cudaMalloc((void **)&d_dest_transpose, size);
        double *transposed_Q = (double *)calloc(size, sizeof(double));

        // Copy mem and exec 
        cudaMemcpy(d_source_transpose, A, size, cudaMemcpyHostToDevice);
        gpuTranspose(d_source_transpose, d_dest_transpose, row, column);
        cudaDeviceSynchronize();
        cudaMemcpy(transposed_Q, d_dest_transpose, size, cudaMemcpyDeviceToHost);
        
        // Free cuda mem
        cudaFree(d_source_transpose);
        cudaFree(d_dest_transpose);

        // End of transpose call
        return transposed_Q;
}

__host__ double *calling_dot_kernel(double *A, double *B, size_t A_row, size_t A_col, size_t B_row, size_t B_col)
{
    size_t sizeA = A_row * A_col * sizeof(float);
    size_t sizeB = B_row * B_col * sizeof(float);
    size_t sizeC = A_row * B_col * sizeof(float);

    float *h_C = (float *)calloc(sizeC, sizeof(float));

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    matrixMultiplication((float*)d_A, (float*)d_B, d_C, A_row, A_col, B_row, B_col, A_row, B_col);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return (double *)h_C;
}

__host__ double *compute_cross_variance_cpu_call_gpu(double *P, double *Q, std::vector<std::tuple<size_t, int>> correspondences, size_t P_r, size_t P_c,
                                size_t Q_r, size_t Q_c) //set default function to lambda function??
{
    UNUSED(Q_r);
    double *cov = (double *)calloc(9, sizeof(double));

    for (auto tup : correspondences)
    {
        auto i = std::get<0>(tup);
        auto j = std::get<1>(tup);
        double *q_point = Q + j * Q_c;
        double *p_point = P + i * P_c;

        double *doted_points = nullptr;
        
        double *transposed_Q = calling_transpose_kernel(q_point, 1, Q_c);
        //double *transposed_Q = transpose(q_point, 1, Q_c);

        dot_product(&doted_points, transposed_Q, p_point, Q_c, 1, 1, P_c); //dim of Q_r * P_r
        free (transposed_Q); 
        increment_cov(cov, doted_points); //need to set element_wise_op but too complicated, doesn't work for some reason.
        free(doted_points);
    }
    return cov;
}

/* -------------- Version GPU Kernel -----------*/

// Implementation with double arrays and no vector for full GPU usage
__global__ void get_correspondence_indices_array_gpu(tuple **correspondances, double *P, double *Q, size_t P_r, size_t P_c, size_t Q_r, size_t Q_c)
{
    int push_index = 0;
    for (size_t i = 0; i < P_r; i++)
    {
        double *p_point = P + i * P_c;
        double min_dist = std::numeric_limits<double>::max();
        int chosen_idx = -1;
        for (size_t j = 0; j < Q_r; j++)
        {
            double *q_point = Q + j * Q_c;
            double dist = std::sqrt(*p_point + *q_point);
            //double dist = std::sqrt(element_wise_reduce(p_point, q_point, 1, P_c, 1, Q_c,
            //                        squared_norm_2, add, add)); //norm 2 between 2 vectors
            if (dist < min_dist)
            {
                min_dist = dist;
                chosen_idx = j;
            }
        }
        tuple *new_tup = nullptr;
        cudaMalloc(&new_tup, sizeof(tuple));
        //tuple *new_tup = (tuple*)calloc(1, sizeof(tuple));
        new_tup->index = i;
        new_tup->value = chosen_idx;
        correspondances[push_index] = new_tup;
        push_index++;
    }
}

// Array implementation for GPU
void compute_cross_variance_array(double * cov, double *P, double *Q, std::tuple<size_t, int> *correspondences, size_t P_r, size_t P_c,
                                size_t Q_r, size_t Q_c) //set default function to lambda function??
{
    UNUSED(Q_r);
    UNUSED(P_r);

    for (size_t index = 0; index < P_r; index ++)
    {
        auto i = std::get<0>(correspondences[index]);
        auto j = std::get<1>(correspondences[index]);
        double *q_point = Q + j * Q_c;
        double *p_point = P + i * P_c;

        double *transposed_Q = transpose(q_point, 1, Q_c);
        double *doted_points = nullptr;
        dot_product(&doted_points, transposed_Q, p_point, Q_c, 1, 1, P_c); //dim of Q_r * P_r
        free (transposed_Q); 
        increment_cov(cov, doted_points); //need to set element_wise_op but too complicated, doesn't work for some reason.
        free(doted_points);
    }
}

__global__ void naiveGPUTranspose(const double *d_a, double *d_b, const int rows, const int cols) 
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int index_in = i * cols + j;
    int index_out = j * rows + i;

    if (i < rows && j < cols)
        d_b[index_out] = d_a[index_in];
}

void gpuTranspose(double* A, double* B, int numRows, int numColumns) {

    // declare the number of blocks per grid and the number of threads per block
    dim3 threadPerBlock(Tile_size, Tile_size);//Number of threads in each block
    dim3 numBlocks((numColumns/ Tile_size) + 1, (numRows/ Tile_size) + 1);//Number of Blocks required

    //@@ Launch the GPU Kernel here
    naiveGPUTranspose<<<numBlocks, threadPerBlock>>>(A, B, numRows, numColumns);
}
