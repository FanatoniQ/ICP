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

/** basic_operations (put this in basic_operations.cpp and co)
 ** TODO: add this to basiq_operations.cpp with ifdef
 **

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
**/

/**
__device__
double add2(double a, double b)
{
    return a + b;
}

__device__
double subtract2(double a, double b)
{
    return a - b;
}

__device__
double mult2(double a, double b)
{
    return a * b;
}

__device__
double divide2(double a, double b)
{
    return a / b;
}**/

/**
template <typename T>
__device__ func2_t<T> add2_op = add<T>;

template <typename T>
__device__ func2_t<T> subtract2_op = subtract<T>;

template <typename T>
__device__ func2_t<T> mult2_op = mult<T>;

template <typename T>
__device__ func2_t<T> divide2_op = divide<T>;
**/

/** Kernel **/

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
    printf("d_A[%lu,%lu] = %lf + d_B[%lu,%lu] = %lf \n",  idy % a_0, idx % a_1, d_A[(idx % a_1) + d_apitch * (idy % a_0)],
		    idy % b_0, idx % b_1, d_B[(idx % b_1) + d_bpitch * (idy % b_0)]);
    BROADCAST_A_B(d_A, d_B, d_R, a_0, a_1, d_apitch, b_0, b_1, d_bpitch, r_0, r_1, d_rpitch, idx, idy, -);
    printf("d_R[%lu,%lu] = %lf \t", idy, idx, d_R[idx + d_rpitch * idy]);

    /**
    assert(d_apitch == d_rpitch);
    // % is slow, have optimized versions without broadcast
    printf("d_A[%lu,%lu] = %lf + d_B[%lu,%lu] = %lf \n",  idy % a_0, idx % a_1, d_A[(idx % a_1) + d_apitch * (idy % a_0)],
		    idy % b_0, idx % b_1, d_B[(idx % b_1) + d_bpitch * (idy % b_0)]);
    d_R[idx + d_rpitch * idy] = d_A[(idx % a_1) + d_apitch * (idy % a_0)] - d_B[(idx % b_1) + d_bpitch * (idy % b_0)];
    printf("d_R[%lu,%lu] = %lf \t", idy, idx, d_R[idx + d_rpitch * idy]); 
    **/
}

/**
 ** \brief broadcast_op_kernel performs numpy style broadcasting on the given matrices:
 ** d_R = d_A op d_B, with the given op
 ** can be launched with <<<blocks,threads>>> as long as blocks*threads >= r_0 * r_1
 ** otherwise the data will not be entirely processed
 ** \note benchmark device memory access... we only need one access per elements, but
 ** this implementation will surely benefit from shared_memory for (vectors / scalars operands)
 **
 ** \param d_A the a_0 x a_1 left operand matrix
 ** \param d_B the b_0 x b_1 right operand matrix
 ** \param d_R the r_0 x r_1 result matrix
 ** \param op the __device__ function 2 operands operation to be used on each
 ** elements from d_A and d_B
 ** \param a_0 the number of lines in d_A
 ** \param a_1 the number of columns in d_A
 ** \param d_apitch the pitch of d_A NOT in bytes
 ** \param b_0 the number of line in d_B
 ** \param b_1 the number of columns in d_B
 ** \param d_bpitch the pitch of d_B NOT in bytes
 ** \param r_0 the number of line in d_R
 ** \param r_1 the number of columns in d_R
 ** \param d_rpitch the pitch of d_R NOT in bytes
 **
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
}**/

// explicit instanciation for lib import

/**
template
__global__ void broadcast_op_kernel<double>(const double *d_A, double *d_B, double *d_R, func2_t<double> op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch);
**/

/**
template
__device__ double add2<double>(double a,double b);

template
__device__ double subtract2<double>(double a, double b);

template
__device__ double mult2<double>(double a, double b);

template
__device__ double divide2<double>(double a, double b);
**/






/**
template <typename T>
__host__ __device__ func2_t<T> add2_op = add<T>;

template <typename T>
__host__ __device__ func2_t<T> subtract2_op = subtract<T>;

template <typename T>
__host__ __device__ func2_t<T> mult2_op = mult<T>;

template <typename T>
__host__ __device__ func2_t<T> divide2_op = divide<T>;
**/

/**
template <typename T>
__device__ func2_t<T> add2_op = add<T>;

template <typename T>
__device__ func2_t<T> subtract2_op = subtract<T>;

template <typename T>
__device__ func2_t<T> mult2_op = mult<T>;

template <typename T>
__device__ func2_t<T> divide2_op = divide<T>;
**/

/**
template
__device__ func2_t<double> add2_op<double>;

template
__device__ func2_t<double> subtract2_op<double>;

template
__device__ func2_t<double> mult2_op<double>;

template
__device__ func2_t<double> divide2_op<double>;
**/
