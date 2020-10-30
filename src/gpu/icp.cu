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
#include "libgpualg/euclidist.cuh"
#include "error.cuh"
#include "libgpualg/mean.cuh"
#include "libgpualg/ope.cuh"
#include "libgpualg/svd.cuh"

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
    size_t sizeA = A_row * A_col * sizeof(double);
    size_t sizeB = B_row * B_col * sizeof(double);
    size_t sizeC = A_row * B_col * sizeof(double);

    double *h_C = (double *)calloc(sizeC, sizeof(double));

    double *d_A;
    double *d_B;
    double *d_C;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);


    matrixMultiplication(d_A, d_B, d_C, A_row, A_col, B_row, B_col, A_row, B_col);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return h_C;
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

dim3 get_gridsize(size_t a_0, size_t a_1, size_t b_0, size_t b_1)
{
    size_t r_0, r_1;
    get_broadcastable_size(a_0, a_1, b_0, b_1, &r_0, &r_1);
    int nbblocksx = std::ceil((float)r_1 / blocksize.x);
    int nbblocksy = std::ceil((float)r_0 / blocksize.y);
    return dim3(nbblocksx, nbblocksy);
}

void icp()
{
    CPUMatrix P, Q;
    //----- MALLOC -----/
    /*
    cudaMalloc(Q_center) dim(Q.dim1)
    cudaMalloc(Q_centered) dim(Q.dim0 * Q.dim1)
    cudaMalloc(P_copy) // the size won't change
    cudaMalloc(P_centered) dim(P.dim0 * P.dim1)
    cudaMalloc(P_center) (axis = 0) (sizeof * dim1)?
    cudaMalloc(cross_var) (3*3) aka (dim1 * dim1)
    cudaMalloc(U) and V_T ? S is not used
    // U dim(cov.dim0 * cov.dim0) and V (cov.dim1 * cov.dim1)
    cudaMalloc(R) rotation matrix dim(U.dim0 * VT.dim1)
    cudaMalloc(t) translation matrix dim(Qcenter.Dim0 * Qcenter.dim1)
    */
    // Device pointers
    double* dQ_center, *dQ_centered, 
        *dP_copy, *dP_centered,*dP_center,
        *dDot_temp,
        *dcorresps, *dcross_var, 
        *dU, *dS, *dV_T, 
        *dR, *dR_transpose, *dt;

    cudaMalloc(&dQ_center, Q.getDim1() * sizeof(double));
    cudaMalloc(&dQ_centered, Q.getDim0() * Q.getDim1() * sizeof(double));
    cudaMalloc(&dP_copy, P.getDim0() * P.getDim1() * sizeof(double));
    cudaMalloc(&dP_centered, P.getDim0() * P.getDim1() * sizeof(double));
    cudaMalloc(&dP_center, P.getDim1() * sizeof(double));
    cudaMalloc(&dDot_temp, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dcross_var, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dU, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dS, P.getDim1() * P.getDim1() * sizeof(double)); // FIXME is it rly the good shape
    cudaMalloc(&dV_T, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dR, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dR_transpose, P.getDim1() * P.getDim1() * sizeof(double)); // FIXME, can use dDot_temp in replacement,
    cudaMalloc(&dt, Q.getDim1() * sizeof(double)); // FIXME
    // cudaMallocPitch(dcorresps) dim(Plines * 1)

    //----- MEMCPY -----/
    cudaMemcpy(dQ_centered, Q.getArray(), Q.getDim0() * Q.getDim1() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dP_copy, P.getArray(), P.getDim0() * P.getDim1() * sizeof(double), cudaMemcpyHostToDevice);

    size_t reducepitch;
    size_t threads_num = 4;
    // Center data P and Q
    // Q_center cuda malloc and mean
    //// auto Q_center = Q.mean(0);
    // Move Q to device and call it Q_centered, apply Q_centered = Q_centered - Q_center
    //// Q -= Q_center;

    //------COMPUTATION------/
    // pitch = dim1 * sizeof()
    // Mean Q_center = Q.mean(0)
    mean_0(dQ_centered, dQ_center, Q.getDim0(), Q.getDim1(), Q.getDim1() * sizeof(double), &reducepitch, threads_num);
    // FIXME pass pointer or reference dQ_center ?

    // Subtract Q -= Q_center
    dim3 blocksize(32, 32);
    auto gridsize = get_gridsize(Q.getDim0(), Q.getDim1(), 1, Q.getDim1());
    matrix_op<double>(gridsize, blocksize, dQ_centered, dQ_center, dQ_centered, MatrixOP::SUBTRACT, 
        Q.getDim0(), Q.getDim1(), Q.getDim1() * sizeof(double), 
        1, Q.getDim1(), Q.getDim1() * sizeof(double), 
        Q.getDim0(), Q.getDim1(), Q.getDim1() * sizeof(double));

    ////std::vector<std::tuple<size_t, int>> correps_values; // Might need device to host move in for loop
    ////std::vector<double> norm_values; // Might need device to host move in for loop
    ////CPUMatrix P_copy; // No CPUMatrix but array of double
    ////P_copy = P; // CUDA malloc of size P both P and P_copy
    // cuda memcpy device to device to put equal P_centered and P_copy
    for (unsigned i = 0; i < iterations; ++i)
    {
        //// auto P_center = P_copy.mean(0);
        // Mean calculation, pass P_center pointer directly as result
        mean_0(dP_copy, dP_center, P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double), &reducepitch, threads_num);
        // FIXME pass pointer or reference ?

        // Center P
        //// P = P_copy - P_center;
        // Substract and put result in P_centered
        // but first compute new gridsize
        gridsize = get_gridsize(P.getDim0(), P.getDim1(), 1, P.getDim1());
        matrix_op<double>(gridsize, blocksize, dP_copy, dP_center, dP_centered, MatrixOP::SUBTRACT,
            P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double),
            1, P.getDim1(), P.getDim1() * sizeof(double),
            P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double));

        // Compute correspondences indices
        auto corresps = get_correspondence_indices(P, Q);
        // Call correspondence indices gpu with (P_centered, Q_centered)
        // use dcorresps

        correps_values.insert(correps_values.end(), corresps.begin(), corresps.end());
        // Insert or not? If we do we have to move back to host
        norm_values.push_back(P.euclidianDistance(Q));
        // call GPU euclidian distance? Move back to host
        // same
        auto cross_var = compute_cross_variance(P, Q, corresps, default_kernel);
        // Compute cross var GPU, call with (P_centered, Q_centered, corresps, default_kernel)
        // 
        // cross_var is here 3*3 mat
        // U, S, V_T = svd
        /*
        auto [U, S, V_T] = std::get<0>(cross_var).svd();
        std::cout << "U: \n"
            << U << std::endl;
        std::cout << "S: \n"
            << S << std::endl;
        std::cout << "V_T: \n"
            << V_T << std::endl;
        UNUSED(S); // unused
        */
        svd_gpu(dcross_var, P.getDim1(), P.getDim1(), dU, dS, dV_T); // FIXME check size
        // Rotation matrix
        //// auto R = U.dot(V_T);
        gridsize = get_gridsize(P.getDim1(), P.getDim1(), P.getDim1(), P.getDim1());
        matrix_op<double>(gridsize, blocksize, dU, dV_T, dR, MatrixOP::MULT,
            P.getDim1(), P.getDim1(), P.getDim1() * sizeof(double),
            P.getDim1(), P.getDim1(), P.getDim1() * sizeof(double),
            P.getDim1(), P.getDim1(), P.getDim1() * sizeof(double));

        // cudaMalloc(R)? rotation
        // Translation Matrix
        auto t = Q_center - P_center.dot(R.transpose());
        // 3 different calculations
        // transpose
        gpuTranspose(dR, dR_transpose, P.getDim1(), P.getDim1());
        // dot product
        // Normally dt should fit the right dimension
        gridsize = get_gridsize(1, P.getDim1(), P.getDim1(), P.getDim1());
        matrix_op<double>(gridsize, blocksize, dP_center, dR_transpose, dt, MatrixOP::MULT,
            1, P.getDim1(), P.getDim1() * sizeof(double),
            P.getDim1(), P.getDim1(), P.getDim1() * sizeof(double),
            1, P.getDim1(), P.getDim1() * sizeof(double));
        // subtract
        gridsize = get_gridsize(1, Q.getDim1(), 1, P.getDim1());
        matrix_op<double>(gridsize, blocksize, dQ_center, dt, dt, MatrixOP::SUBTRACT,
            1, Q.getDim1(), Q.getDim1() * sizeof(double),
            1, P.getDim1(), P.getDim1() * sizeof(double),
            1, P.getDim1(), P.getDim1() * sizeof(double));
        // Update P
        P_copy = P_copy.dot(R.transpose()) + t;
        // use same device pointer for the dot product both dimensions being the same
        // first transpose - already done with R transpose
        // dot product / use P_centered to store the result bc no need of the data anymore
        gridsize = get_gridsize(P.getDim0(), P.getDim1(), P.getDim1(), P.getDim1());
        matrix_op<double>(gridsize, blocksize, dP_copy, dR_transpose, dP_centered, MatrixOP::MULT,
            P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double),
            P.getDim1(), P.getDim1(), P.getDim1() * sizeof(double),
            P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double));
        // plus
        gridsize = get_gridsize(P.getDim0(), P.getDim1(), 1, P.getDim1());
        matrix_op<double>(gridsize, blocksize, dP_centered, dt, dP_copy, MatrixOP::ADD,
            P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double),
            1, P.getDim1(), P.getDim1() * sizeof(double),
            P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double));
    }
    correps_values.push_back(correps_values.back());


    cudaFree(dQ_center);
    cudaFree(dQ_centered);
    cudaFree(dP_copy);
    cudaFree(dP_centered);
    cudaFree(dP_center);
    cudaFree(dDot_temp);
    cudaFree(dcross_var);
    cudaFree(dU);
    cudaFree(dV_T);
    cudaFree(dR);
    cudaFree(dR_transpose);
    cudaFree(dt);

    //return std::make_tuple(std::move(P_copy), norm_values, correps_values);
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
