#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <assert.h>

// CPU
#include "libCSV/csv.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/alg.hpp"
#include "libalg/print.hpp"
#include "libalg/broadcasting.hpp"
#include "error.hpp"

// GPU
#include "error.cuh"
#include "libgpualg/ope.cuh"

template <typename T> 
__device__
T add2(T a, T b)
{
    return a + b;
}

template <typename T> 
__device__
T subtract2(T a, T b)
{
    return a - b;
}

template <typename T> 
__device__
T mult2(T a, T b)
{
    return a * b;
}

template <typename T> 
__device__
T divide2(T a, T b)
{
    return a / b;
}

// explicit pointer instanciation for use in kernel...
// TODO: export this in static lib, was linking failing or invalid device pointer
// we could use constant memory function table array in static lib for exemple

template <typename T>
__device__ func2_t<T> add2_op = add2<T>;

template <typename T>
__device__ func2_t<T> subtract2_op = subtract2<T>;

template <typename T>
__device__ func2_t<T> mult2_op = mult2<T>;

template <typename T>
__device__ func2_t<T> divide2_op = divide2<T>;


/** Kernel **/

// TODO: remove me: deprecated
#define BROADCAST_A_B(d_A, d_B, d_R, a_0, a_1, d_apitch, b_0, b_1, d_bpitch, r_0, r_1, d_rpitch, idx, idy, op) {\
    d_R[idx + d_rpitch * idy] = d_A[(idx % a_1) + d_apitch * (idy % a_0)] op d_B[(idx % b_1) + d_bpitch * (idy % b_0)]; }\

/** basic version
 ** FIXME: remove this in profit of broadcast_op_kernel onces working
 ** WATCH out: pitch is NOT in bytes
 **/
__global__ void broadcast_subtract_kernel(const double *d_A, double *d_B, double *d_R,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // column
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y; // line
    if (idy >= r_0 || idx >= r_1)
        return;
    assert(d_apitch == d_rpitch);
    // printf("d_A[%lu,%lu] = %lf + d_B[%lu,%lu] = %lf \n",  idy % a_0, idx % a_1, d_A[(idx % a_1) + d_apitch * (idy % a_0)],
	//	    idy % b_0, idx % b_1, d_B[(idx % b_1) + d_bpitch * (idy % b_0)]);
    BROADCAST_A_B(d_A, d_B, d_R, a_0, a_1, d_apitch, b_0, b_1, d_bpitch, r_0, r_1, d_rpitch, idx, idy, -);
    // printf("d_R[%lu,%lu] = %lf \t", idy, idx, d_R[idx + d_rpitch * idy]);
}

template <typename T>
__global__ void broadcast_op_kernel(const T *d_A, T *d_B, T *d_R, func2_t<T> op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; // column
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y; // line
    if (idy >= r_0 || idx >= r_1)
        return;
    // % is slow, have optimized versions without broadcast
    d_R[idx + d_rpitch * idy] = (*op)(d_A[(idx % a_1) + d_apitch * (idy % a_0)], d_B[(idx % b_1) + d_bpitch * (idy % b_0)]);
}

template <typename T>
__global__ void broadcast_op_line_vector_kernel(const T *d_A, T *d_B, T *d_R, func2_t<T> op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; // column
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y; // line
//    extern __shared__ T s_vector[]; // double to store vector of size (blockDim.x)
    assert((b_0 == 1) && "d_B should be a line vector !");
    assert((a_0 == r_0 && a_1 == r_1) && "Invalid shape for line vector op resulting matrix");
    if (idy >= r_0 || idx >= r_1)
        return;
//    if (threadIdx.y == 0) // first line of block loads vector
//        s_vector[threadIdx.x] = d_B[idx]; // b_0 == 1
//    __syncthreads(); // wait for vector to be avalaible for all threads in block
    d_R[idx + d_rpitch * idy] = (*op)(d_A[idx + d_apitch * idy], d_B[idx]);
}

template <typename T>
__global__ void broadcast_op_column_vector_kernel(const T *d_A, T *d_B, T *d_R, func2_t<T> op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; // column
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y; // line
//    extern __shared__ T s_vector[]; // double to store vector of size (blockDim.y)
    assert((b_1 == 1) && "d_B should be a column vector !");
    assert((a_0 == r_0 && a_1 == r_1) && "Invalid shape for line vector op resulting matrix");
    if (idy >= r_0 || idx >= r_1)
        return;
//    if (threadIdx.x == 0) // first column of block loads vector
//        s_vector[threadIdx.y] = d_B[idy * d_bpitch];
//    __syncthreads(); // wait for vector to be avalaible for all threads in block
    d_R[idx + d_rpitch * idy] = (*op)(d_A[idx + d_apitch * idy], d_B[idy * d_bpitch]);
}


template <typename T>
__global__ void broadcast_op_scalar_kernel(const T *d_A, T *d_B, T *d_R, func2_t<T> op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; // column
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y; // line
//    __shared__ T s_scalar[1]; // double to store scalar
    assert((a_0 == r_0 && a_1 == r_1) && "Invalid shape for scalar op resulting matrix");
    if (idy >= r_0 || idx >= r_1)
        return;
//    if (threadIdx.x == 0 && threadIdx.y == 0)
//        s_scalar[0] = d_B[0];
//    __syncthreads(); // wait for scalar to be avalaible for all threads in block
    d_R[idx + d_rpitch * idy] = (*op)(d_A[idx + d_apitch * idy], d_B[0]);
}

template <typename T>
__host__ void matrix_op(dim3 gridsize, dim3 blocksize, 
    const T *d_A, T *d_B, T *d_R, enum MatrixOP op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch)
{
    func2_t<T> h_op;
    if (op == MatrixOP::ADD)
        cudaMemcpyFromSymbol(&h_op, add2_op<T>, sizeof(func2_t<T>));
    else if (op == MatrixOP::SUBTRACT)
        cudaMemcpyFromSymbol(&h_op, subtract2_op<T>, sizeof(func2_t<T>));
    else if (op == MatrixOP::MULT)
        cudaMemcpyFromSymbol(&h_op, mult2_op<T>, sizeof(func2_t<T>));
    else if (op == MatrixOP::DIVIDE)
        cudaMemcpyFromSymbol(&h_op, divide2_op<T>, sizeof(func2_t<T>));
    else
    {
        fprintf(stderr, "Invalid Operation argument for matrix_add !");
        return;
    }
    cudaCheckError();
    if (b_0 == 1 && b_1 == 1) {
        broadcast_op_scalar_kernel<T><<<gridsize, blocksize>>>(d_A, d_B, d_R, h_op,
            a_0, a_1, d_apitch / sizeof(T),
            r_0, r_1, d_rpitch / sizeof(T));
    } else if (b_0 == 1) {
        broadcast_op_line_vector_kernel<T><<<gridsize, blocksize>>>(d_A, d_B, d_R, h_op,
            a_0, a_1, d_apitch / sizeof(T),
            b_0, b_1, d_bpitch / sizeof(T),
            r_0, r_1, d_rpitch / sizeof(T));
    } else if (b_1 == 1) {
        broadcast_op_column_vector_kernel<T><<<gridsize, blocksize>>>(d_A, d_B, d_R, h_op,
            a_0, a_1, d_apitch / sizeof(T),
            b_0, b_1, d_bpitch / sizeof(T),
            r_0, r_1, d_rpitch / sizeof(T));
    } else {
        broadcast_op_kernel<T><<<gridsize, blocksize>>>(d_A, d_B, d_R, h_op,
            a_0, a_1, d_apitch / sizeof(T),
            b_0, b_1, d_bpitch / sizeof(T),
            r_0, r_1, d_rpitch / sizeof(T));
    }
    cudaDeviceSynchronize();
    cudaCheckError();
};

// explicit instanciation for lib import

template
__global__ void broadcast_op_kernel<double>(const double *d_A, double *d_B, double *d_R, func2_t<double> op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch);

template
__global__ void broadcast_op_scalar_kernel<double>(const double *d_A, double *d_B, double *d_R, func2_t<double> op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch);

template
__global__ void broadcast_op_line_vector_kernel<double>(const double *d_A, double *d_B, double *d_R, func2_t<double> op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch);

template
__global__ void broadcast_op_column_vector_kernel<double>(const double *d_A, double *d_B, double *d_R, func2_t<double> op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch);

template
__host__ void matrix_op<double>(dim3 gridsize, dim3 blocksize,
    const double *d_A, double *d_B, double *d_R, enum MatrixOP op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch);
