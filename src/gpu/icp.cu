#include <vector>
#include <limits>
#include <tuple>
#include <iostream>
#include <cmath>

#include "libalg/basic_operations.hpp"
#include "libalg/alg.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/CPUView.hpp"
#include "libalg/broadcasting.hpp"
#include "error.hpp"
#include "cpu/tuple.hpp"

#include "gpu/icp.cuh"
#include "libgpualg/mult.cuh"
#include "libgpualg/euclidist.cuh"
#include "error.cuh"
#include "libgpualg/mean.cuh"
#include "libgpualg/ope.cuh"
#include "libgpualg/svd.cuh"
#include "libgpuicp/dist.cuh"
#include "libgpuicp/batchcovs.cuh"

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
            if (dist < min_dist)
            {
                min_dist = dist;
                chosen_idx = j;
            }
        }
        tuple *new_tup = nullptr;
        cudaMalloc(&new_tup, sizeof(tuple));
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

dim3 get_gridsize(size_t a_0, size_t a_1, size_t b_0, size_t b_1, dim3 blocksize)
{
    size_t r_0, r_1;
    get_broadcastable_size(a_0, a_1, b_0, b_1, &r_0, &r_1);
    int nbblocksx = std::ceil((float)r_1 / blocksize.x);
    int nbblocksy = std::ceil((float)r_0 / blocksize.y);
    return dim3(nbblocksx, nbblocksy);
}

// TODO: REMOVE ME since useless
__global__ void print_matrix_kern(char* d_A, int pitch, int nbvals)
{
    int j;
    int idx = threadIdx.x;
    double* line = (double*)(d_A + idx * pitch);
    printf("Line %d:\n", idx);
    __syncthreads();
    for (j = 0; j < nbvals; ++j) {
        //printf("%6.2f\t", (double)(d_A[idx * pitch + j * sizeof(double)]));
        printf("%6.2f\t", line[j]);
        __syncthreads();
    }
    printf("\n");
    __syncthreads();
}

void print_Mat_gpu(double *dmat, int m, int n, const char* name)
{
    double* Mat = (double*)malloc(m * n * sizeof(double));
    cudaMemcpy(Mat, dmat, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            double Areg = Mat[col + row * n];
            printf("%s(%d,%d) = %f ", name, row, col, Areg);
        }
        printf("\n");
    }
    free(Mat);
}

void print_Mat_gpu(unsigned int* dmat, int m, int n, const char* name)
{
    unsigned int* Mat = (unsigned int*)malloc(m * n * sizeof(unsigned int));
    cudaMemcpy(Mat, dmat, m * n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            unsigned int Areg = Mat[col + row * n];
            printf("%s(%d,%d) = %u ", name, row, col, Areg);
        }
        printf("\n");
    }
    free(Mat);
}

void print_corresp_gpu(ICPCorresp* dmat, int m, int n, const char* name)
{
    ICPCorresp* Mat = (ICPCorresp*)malloc(m * n * sizeof(ICPCorresp));
    cudaMemcpy(Mat, dmat, m * n * sizeof(ICPCorresp), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            ICPCorresp Areg = Mat[col + row * n];
            printf("%s(%d,%d) = (%f,%d) ", name, row, col, Areg.dist, Areg.id);
        }
        printf("\n");
    }
    free(Mat);
}

CPUMatrix icp_gpu(CPUMatrix& P, CPUMatrix& Q, unsigned iterations)
{
    // Assuming most of the time P.getdim1() == Q.getdim1()
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
    cudaMalloc(corresps) dim(P
    */
    // Device pointers
    double* dQ_center, *dQ_centered, 
        *dP_copy, *dP_centered,*dP_center,
        *dDot_temp, 
        *dU, *dS, *dV_T, 
        *dR, *dR_transpose, *dt;

    // Corresps device pointers
    ICPCorresp* dcorresps;
    double* dcross_var;
    double* d_R;

    size_t dcorresps_pitch;
    size_t cross_var_pitch = P.getDim1() * Q.getDim1() * sizeof(double);
    size_t reducepitch = Q.getDim1() * sizeof(double);
    size_t r_pitch = P.getDim1() * Q.getDim1() * sizeof(double);
    size_t cov_pitch = P.getDim1() * Q.getDim1() * sizeof(double);

    size_t threads_num = 1024;
    size_t batchsize = 16;

    std::cerr << "==== Init ====" << std::endl;
    dQ_center = nullptr; // reduce_0 function does the allocation if nullptr
    cudaMalloc(&dQ_centered, Q.getDim0() * Q.getDim1() * sizeof(double));
    cudaMalloc(&dP_copy, P.getDim0() * P.getDim1() * sizeof(double));
    cudaMalloc(&dP_centered, P.getDim0() * P.getDim1() * sizeof(double));
    dP_center = nullptr; // reduce_0 function does the allocation if nullptr
    cudaMalloc(&dDot_temp, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dU, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dS, P.getDim1() * P.getDim1() * sizeof(double)); // FIXME shape?
    cudaMalloc(&dV_T, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dR, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dR_transpose, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dt, Q.getDim1() * sizeof(double));

    cudaMallocPitch((void**)&dcorresps, &dcorresps_pitch, Q.getDim0() * sizeof(ICPCorresp), batchsize);
    cudaCheckError();
    cudaMalloc((void**)&d_R, batchsize * r_pitch);
    cudaCheckError();
    cudaMalloc((void**)&dcross_var, 1 * cov_pitch);
    cudaCheckError();

    //----- MEMCPY -----/
    cudaMemcpy(dQ_centered, Q.getArray(), Q.getDim0() * Q.getDim1() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dP_copy, P.getArray(), P.getDim0() * P.getDim1() * sizeof(double), cudaMemcpyHostToDevice);

    // Center data P and Q
    // Q_center cuda malloc and mean
    // Move Q to device and call it Q_centered, apply Q_centered = Q_centered - Q_center

    //------COMPUTATION------/
    // pitch = dim1 * sizeof()
    // Mean Q_center = Q.mean(0)
    reduce_0(MatrixReduceOP::MEAN, dQ_centered, &dQ_center, Q.getDim1(), Q.getDim0(), Q.getDim1() * sizeof(double), &reducepitch, threads_num);

    // Subtract Q -= Q_center
    dim3 blocksize(32, 32);
    auto gridsize = get_gridsize(Q.getDim0(), Q.getDim1(), 1, Q.getDim1(), blocksize);
    matrix_op<double>(gridsize, blocksize, dQ_centered, dQ_center, dQ_centered, MatrixOP::SUBTRACT, 
        Q.getDim0(), Q.getDim1(), Q.getDim1() * sizeof(double), 
        1, Q.getDim1(), Q.getDim1() * sizeof(double), 
        Q.getDim0(), Q.getDim1(), Q.getDim1() * sizeof(double));

    // cuda memcpy device to device to put equal P_centered and P_copy
    for (unsigned i = 0; i < iterations; ++i)
    {
        // Mean calculation, pass P_center pointer directly as result
        mean_0(dP_copy, &dP_center, P.getDim1(), P.getDim0(), P.getDim1() * sizeof(double), &reducepitch, threads_num);

        // Center P
        // Substract and put result in P_centered
        // but first compute new gridsize
        gridsize = get_gridsize(P.getDim0(), P.getDim1(), 1, P.getDim1(), blocksize);
        matrix_op<double>(gridsize, blocksize, dP_copy, dP_center, dP_centered, MatrixOP::SUBTRACT,
            P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double),
            1, P.getDim1(), P.getDim1() * sizeof(double),
            P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double));

        // Compute correspondences indices
        // Call correspondence indices gpu with (P_centered, Q_centered)
        // Compute cross var GPU, call with (P_centered, Q_centered, corresps, default_kernel)
        get_batch_cov(dP_centered, P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double),
            dQ_centered, Q.getDim0(), Q.getDim1(), Q.getDim1() * sizeof(double),
            dcorresps, batchsize, Q.getDim0(), dcorresps_pitch,
            d_R, batchsize, P.getDim1() * Q.getDim1(), r_pitch,
            dcross_var, P.getDim1(), Q.getDim1(), cov_pitch,
            batchsize
        );
        // cross_var is here 3*3 mat

        svd_gpu(dcross_var, P.getDim1(), P.getDim1(), dV_T, dS, dU);

        // Rotation matrix
        matrixMultiplication(dU, dV_T, dR,
            P.getDim1(), P.getDim1(),
            P.getDim1(), P.getDim1(),
            P.getDim1(), P.getDim1());

        // Translation Matrix
        // 3 different calculations
        // transpose
        gpuTranspose(dR, dR_transpose, P.getDim1(), P.getDim1());
        // dot product
        // Normally dt should fit the right dimension
        matrixMultiplication(dP_center, dR_transpose, dt,
            1, P.getDim1(),
            P.getDim1(), P.getDim1(),
            1, P.getDim1());
        // subtract
        gridsize = get_gridsize(1, Q.getDim1(), 1, P.getDim1(), blocksize);
        matrix_op<double>(gridsize, blocksize, dQ_center, dt, dt, MatrixOP::SUBTRACT,
            1, Q.getDim1(), Q.getDim1() * sizeof(double),
            1, P.getDim1(), P.getDim1() * sizeof(double),
            1, P.getDim1(), P.getDim1() * sizeof(double));

        // Update P
        // use same device pointer for the dot product both dimensions being the same
        // first transpose - already done with R transpose
        // dot product / use P_centered to store the result bc no need of the data anymore
        matrixMultiplication(dP_copy, dR_transpose, dP_centered,
            P.getDim0(), P.getDim1(),
            P.getDim1(), P.getDim1(),
            P.getDim0(), P.getDim1());
        // plus
        gridsize = get_gridsize(P.getDim0(), P.getDim1(), 1, P.getDim1(), blocksize);
        matrix_op<double>(gridsize, blocksize, dP_centered, dt, dP_copy, MatrixOP::ADD,
            P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double),
            1, P.getDim1(), P.getDim1() * sizeof(double),
            P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double));

    }
    cudaDeviceSynchronize();
    double* P_result = (double*)malloc(P.getDim0() * P.getDim1() * sizeof(double));
    cudaMemcpy(P_result, dP_copy, P.getDim0() * P.getDim1() * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dQ_center);
    cudaFree(dQ_centered);
    cudaFree(dP_copy);
    cudaFree(dP_centered);
    cudaFree(dP_center);
    cudaFree(dDot_temp);
    cudaFree(dcross_var);
    cudaFree(dcorresps);
    cudaFree(d_R);
    cudaFree(dU);
    cudaFree(dV_T);
    cudaFree(dR);
    cudaFree(dR_transpose);
    cudaFree(dt);
    cudaDeviceReset();
    cudaCheckError();

    return CPUMatrix(P_result, P.getDim0(), P.getDim1());
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

CPUMatrix icp_gpu_optimized(CPUMatrix& P, CPUMatrix& Q, unsigned iterations) {
    // Assuming most of the time P.getdim1() == Q.getdim1()
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
cudaMalloc(corresps) dim(P
*/
// Device pointers
    double* dQ_center, * dQ_centered,
        * dP_copy, * dP_centered, * dP_center,
        * dDot_temp,
        * dU, * dS, * dV_T,
        * dR, * dR_transpose, * dt;

    // Corresps device pointers
    unsigned int* dcorresps;
    double* dcross_var = nullptr;
    //double* d_R = nullptr;
    unsigned int d_r0 = P.getDim0(), d_r1 = P.getDim1() * Q.getDim1();

    size_t cross_var_pitch = P.getDim1() * Q.getDim1() * sizeof(double);
    size_t reducepitch = Q.getDim1() * sizeof(double);
    size_t r_pitch;
    size_t cov_pitch = P.getDim1() * Q.getDim1() * sizeof(double);

    size_t threads_num = 1024;

    std::cerr << "==== Init ====" << std::endl;
    dQ_center = nullptr; // reduce_0 function does the allocation if nullptr
    cudaMalloc(&dQ_centered, Q.getDim0() * Q.getDim1() * sizeof(double));
    cudaMalloc(&dP_copy, P.getDim0() * P.getDim1() * sizeof(double));
    cudaMalloc(&dP_centered, P.getDim0() * P.getDim1() * sizeof(double));
    dP_center = nullptr; // reduce_0 function does the allocation if nullptr
    cudaMalloc(&dDot_temp, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dU, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dS, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dV_T, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dR, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dR_transpose, P.getDim1() * P.getDim1() * sizeof(double));
    cudaMalloc(&dt, Q.getDim1() * sizeof(double));

    cudaMalloc((void**)&dcorresps, P.getDim0() * sizeof(unsigned int));
    cudaCheckError();

    //----- MEMCPY -----/
    cudaMemcpy(dQ_centered, Q.getArray(), Q.getDim0() * Q.getDim1() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dP_copy, P.getArray(), P.getDim0() * P.getDim1() * sizeof(double), cudaMemcpyHostToDevice);

    // Center data P and Q
    // Q_center cuda malloc and mean
    // Move Q to device and call it Q_centered, apply Q_centered = Q_centered - Q_center

    //------COMPUTATION------/
    // pitch = dim1 * sizeof()
    // Mean Q_center = Q.mean(0)
    reduce_0(MatrixReduceOP::MEAN, dQ_centered, &dQ_center, Q.getDim1(), Q.getDim0(), Q.getDim1() * sizeof(double), &reducepitch, threads_num);

    // Subtract Q -= Q_center
    dim3 blocksize(32, 32);
    auto gridsize = get_gridsize(Q.getDim0(), Q.getDim1(), 1, Q.getDim1(), blocksize);
    matrix_op<double>(gridsize, blocksize, dQ_centered, dQ_center, dQ_centered, MatrixOP::SUBTRACT,
        Q.getDim0(), Q.getDim1(), Q.getDim1() * sizeof(double),
        1, Q.getDim1(), Q.getDim1() * sizeof(double),
        Q.getDim0(), Q.getDim1(), Q.getDim1() * sizeof(double));

    // cuda memcpy device to device to put equal P_centered and P_copy
    for (unsigned i = 0; i < iterations; ++i)
    {
        // Mean calculation, pass P_center pointer directly as result
        mean_0(dP_copy, &dP_center, P.getDim1(), P.getDim0(), P.getDim1() * sizeof(double), &reducepitch, threads_num);

        // Center P
        // Substract and put result in P_centered
        // but first compute new gridsize
        gridsize = get_gridsize(P.getDim0(), P.getDim1(), 1, P.getDim1(), blocksize);
        matrix_op<double>(gridsize, blocksize, dP_copy, dP_center, dP_centered, MatrixOP::SUBTRACT,
            P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double),
            1, P.getDim1(), P.getDim1() * sizeof(double),
            P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double));

        // Compute correspondences indices
        // Call correspondence indices gpu with (P_centered, Q_centered)
        // Compute cross var GPU, call with (P_centered, Q_centered, corresps, default_kernel)
        get_array_correspondences(dcorresps, dP_centered, dQ_centered, 
            P.getDim0(), P.getDim1(), 
            Q.getDim0(), Q.getDim1());
        //print_Mat_gpu(dcorresps, 1, P.getDim0(), "Csp");
        get_array_cross_covs_flattened(dP_centered, dQ_centered, &dcross_var, dcorresps,
            P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double),
            Q.getDim0(), Q.getDim1(), Q.getDim1() * sizeof(double),
            d_r0, d_r1, &r_pitch,
            P.getDim0(), true);
        reduce_0(MatrixReduceOP::SUM, dcross_var, &dcross_var, (size_t) d_r1, (size_t) d_r0, r_pitch, &r_pitch, threads_num);
        //print_Mat_gpu(dcross_var, P.getDim1(), P.getDim1(), "cov");
        // cross_var is here 3*3 mat

        svd_gpu(dcross_var, P.getDim1(), P.getDim1(), dV_T, dS, dU);

        // Rotation matrix
        matrixMultiplication(dU, dV_T, dR,
            P.getDim1(), P.getDim1(),
            P.getDim1(), P.getDim1(),
            P.getDim1(), P.getDim1());

        // Translation Matrix
        // 3 different calculations
        // transpose
        gpuTranspose(dR, dR_transpose, P.getDim1(), P.getDim1());
        // dot product
        // Normally dt should fit the right dimension
        matrixMultiplication(dP_center, dR_transpose, dt,
            1, P.getDim1(),
            P.getDim1(), P.getDim1(),
            1, P.getDim1());
        // subtract
        gridsize = get_gridsize(1, Q.getDim1(), 1, P.getDim1(), blocksize);
        matrix_op<double>(gridsize, blocksize, dQ_center, dt, dt, MatrixOP::SUBTRACT,
            1, Q.getDim1(), Q.getDim1() * sizeof(double),
            1, P.getDim1(), P.getDim1() * sizeof(double),
            1, P.getDim1(), P.getDim1() * sizeof(double));

        // Update P
        // use same device pointer for the dot product both dimensions being the same
        // first transpose - already done with R transpose
        // dot product / use P_centered to store the result bc no need of the data anymore
        matrixMultiplication(dP_copy, dR_transpose, dP_centered,
            P.getDim0(), P.getDim1(),
            P.getDim1(), P.getDim1(),
            P.getDim0(), P.getDim1());
        // plus
        gridsize = get_gridsize(P.getDim0(), P.getDim1(), 1, P.getDim1(), blocksize);
        matrix_op<double>(gridsize, blocksize, dP_centered, dt, dP_copy, MatrixOP::ADD,
            P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double),
            1, P.getDim1(), P.getDim1() * sizeof(double),
            P.getDim0(), P.getDim1(), P.getDim1() * sizeof(double));

    }
    cudaDeviceSynchronize();
    double* P_result = (double*)malloc(P.getDim0() * P.getDim1() * sizeof(double));
    cudaMemcpy(P_result, dP_copy, P.getDim0() * P.getDim1() * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dQ_center);
    cudaFree(dQ_centered);
    cudaFree(dP_copy);
    cudaFree(dP_centered);
    cudaFree(dP_center);
    cudaFree(dDot_temp);
    cudaFree(dcross_var);
    cudaFree(dcorresps);
    cudaFree(dU);
    cudaFree(dV_T);
    cudaFree(dR);
    cudaFree(dR_transpose);
    cudaFree(dt);
    cudaDeviceReset();
    cudaCheckError();

    return CPUMatrix(P_result, P.getDim0(), P.getDim1());
}
