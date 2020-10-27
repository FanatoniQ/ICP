#pragma once

// template for function pointers
template<typename T>
using func2_t = T (*) (T, T); // type alias quicker

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

/**
 ** \brief broadcast_op_line_vector_kernel performs numpy style broadcasting with the left operand matrix and the given line vector:
 ** d_R = d_A op d_B, with the given op
 ** can be launched with <<<blocks,threads, threads.x * sizeof(T)>>> as long as blocks*threads >= r_0 * r_1
 ** otherwise the data will not be entirely processed. This is an optimized version which uses shared memory instead of
 ** device memory for storing a partial vector line in the block (b_0 == 1) and threfore no broadcasting modulus calculating.
 ** This should have better performance due to modulus but also since we read only nbblocks times each vector line. TODO: benchmark
 **
 ** \param d_A the a_0 x a_1 left operand matrix
 ** \param d_B the b_0 x b_1 right operand vector (b_0 == 1)
 ** \param d_R the r_0 x r_1 result matrix (r_0 == a_1 && r_1 == a_1)
 ** \param op the __device__ function 2 operands operation to be used on each
 ** elements from d_A and d_B
 ** \param a_0 the number of lines in d_A
 ** \param a_1 the number of columns in d_A
 ** \param d_apitch the pitch of d_A NOT in bytes
 ** \param b_0 the number of line in d_B (1)
 ** \param b_1 the number of columns in d_B
 ** \param d_bpitch the pitch of d_B NOT in bytes (not usefull)
 ** \param r_0 the number of line in d_R (== a_0)
 ** \param r_1 the number of columns in d_R (== a_1)
 ** \param d_rpitch the pitch of d_R NOT in bytes
 **/
template <typename T>
__global__ void broadcast_op_line_vector_kernel(const T *d_A, T *d_B, T *d_R, func2_t<T> op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch);

/**
 ** \brief broadcast_op_column_vector_kernel performs numpy style broadcasting with the left operand matrix and the given column vector:
 ** d_R = d_A op d_B, with the given op
 ** can be launched with <<<blocks,threads, threads.y * sizeof(T)>>> as long as blocks*threads >= r_0 * r_1
 ** otherwise the data will not be entirely processed. This is an optimized version which uses shared memory instead of
 ** device memory for storing a partial vector column in the block (rowise in shared memory) (b_1 == 1)
 ** and threfore no broadcasting modulus calculating.
 ** This should have better performance due to modulus but also since we read only nbblocks times each vector line. TODO: benchmark
 **
 ** \param d_A the a_0 x a_1 left operand matrix
 ** \param d_B the b_0 x b_1 right operand vector (b_1 == 1)
 ** \param d_R the r_0 x r_1 result matrix (r_0 == a_1 && r_1 == a_1)
 ** \param op the __device__ function 2 operands operation to be used on each
 ** elements from d_A and d_B
 ** \param a_0 the number of lines in d_A
 ** \param a_1 the number of columns in d_A
 ** \param d_apitch the pitch of d_A NOT in bytes
 ** \param b_0 the number of line in d_B
 ** \param b_1 the number of columns in d_B (1)
 ** \param d_bpitch the pitch of d_B NOT in bytes
 ** \param r_0 the number of line in d_R (== a_0)
 ** \param r_1 the number of columns in d_R (== a_1)
 ** \param d_rpitch the pitch of d_R NOT in bytes
 **/
template <typename T>
__global__ void broadcast_op_column_vector_kernel(const T *d_A, T *d_B, T *d_R, func2_t<T> op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch);

/**
 ** \brief broadcast_op_scalar_kernel performs numpy style broadcasting with the left operand matrix and the given scalar:
 ** d_R = d_A op d_B, with the given op
 ** can be launched with <<<blocks,threads>>> as long as blocks*threads >= r_0 * r_1
 ** otherwise the data will not be entirely processed. This is an optimized version which uses shared memory instead of
 ** device memory for scalar (b_0 == b_1 == 1) and threfore no broadcasting modulus calculating.
 ** \note using the scalar in constant memory (kernel arguments) could be enough for performances.
 ** Benchmark since if the device memory for b is less accessed, only one value so cache should provide good performances
 **
 ** \param d_A the a_0 x a_1 left operand matrix
 ** \param d_B the 1 x 1 right operand matrix (scalar)
 ** \param d_R the r_0 x r_1 result matrix
 ** \param op the __device__ function 2 operands operation to be used on each
 ** elements from d_A and d_B
 ** \param a_0 the number of lines in d_A
 ** \param a_1 the number of columns in d_A
 ** \param d_apitch the pitch of d_A NOT in bytes
 ** \param r_0 the number of line in d_R
 ** \param r_1 the number of columns in d_R
 ** \param d_rpitch the pitch of d_R NOT in bytes
 **/
template <typename T>
__global__ void broadcast_op_scalar_kernel(const T *d_A, T *d_B, T *d_R, func2_t<T> op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch);

/**
 ** \depreacted use broadcast_op_kernel instead
 **/
__global__ void broadcast_subtract_kernel(const double *d_A, double *d_B, double *d_R,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch);
