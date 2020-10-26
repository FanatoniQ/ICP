#pragma once

// template for function pointers
template<typename T>
using func2_t = T (*) (T, T); // type alias quicker

/** basic_operations (put this in basic_operations.cpp and co)
 ** TODO: add this to basiq_operations.cpp with ifdef
 **/

template <typename T> 
__host__ __device__
T add(T a, T b);

template <typename T> 
__host__ __device__
T subtract(T a, T b);

template <typename T> 
__host__ __device__
T mult(T a, T b);

template <typename T> 
__host__ __device__
T divide(T a, T b);

/** static pointers for use in kernel

template <typename T>
__device__ func2_t<T> add2_op;

template <typename T>
__device__ func2_t<T> subtract2_op;

template <typename T>
__device__ func2_t<T> mult2_op;

template <typename T>
__device__ func2_t<T> divide2_op;
**/

/** Kernel **/

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
 **/
template <typename T>
__global__ void broadcast_op_kernel(const T *d_A, T *d_B, T *d_R, func2_t<T> op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch);

__global__ void broadcast_subtract_kernel(const double *d_A, double *d_B, double *d_R,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch);
