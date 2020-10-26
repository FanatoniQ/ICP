#include <stdio.h>
#include <iostream>
#include <iomanip>

// CPU
#include "libCSV/csv.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/alg.hpp"
#include "libalg/print.hpp"
#include "libalg/broadcasting"
#include "error.hpp"

// GPU
#include "error.cuh"

// template for function pointers
template<typename T>
using func2_t = T (*) (T, T); // type alias quicker

/** basic_operations (put this in basic_operations.cpp and co)
 ** TODO: add this to basiq_operations.cpp with ifdef
 **/

template <typename T> 
#ifdef __CUDACC__
__host__ __device__
#endif
T add(T a, T b)
{
    return a + b;
}

template <typename T> 
#ifdef __CUDACC__
__host__ __device__
#endif
T subtract(T a, T b)
{
    return a - b;
}

template <typename T> 
#ifdef __CUDACC__
__host__ __device__
#endif
T mult(T a, T b)
{
    return a * b;
}

template <typename T> 
#ifdef __CUDACC__
__host__ __device__
#endif
T divide(T a, T b)
{
    return a / b;
}

/** static pointers for use in kernel **/

template <typename T>
__device__ func2_t<T> add2_op = add<T>;

template <typename T>
__device__ func2_t<T> subtract2_op = subtract<T>;

template <typename T>
__device__ func2_t<T> mult2_op = mult<T>;

template <typename T>
__device__ func2_t<T> divide2_op = divide<T>;

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
 ** \param d_apitch the pitch of d_A in bytes
 ** \param b_0 the number of line in d_B
 ** \param b_1 the number of columns in d_B
 ** \param d_bpitch the pitch of d_B in bytes
 ** \param r_0 the number of line in d_R
 ** \param r_1 the number of columns in d_R
 ** \param d_rpitch the pitch of d_R in bytes
 **/
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
    d_R[idx + d_rpitch * idy] = op(d_A[(idx % a_1) + d_apitch * (idy % a_0)], d_B[(idx % b_1) + d_bpitch * (idy % b_0)]);
}

int main(int argv, char **argc)
{
    runtime_assert(argc == 2, "Usage: ./testgpuope file1");
    
    // retrieving functions (this part is not required if not on __host__ function)
    func2_t<double> h_add2_op, h_subtract2_op, h_mult2_op, h_divide2_op;
    cudaMemcpyFromSymbol(&h_add2_op, add2_op<double>, sizeof(func2_t<double>));
    cudaMemcpyFromSymbol(&h_subtract2_op, subtract2_op<double>, sizeof(func2_t<double>));
    cudaMemcpyFromSymbol(&h_mult2_op, mult2_op<double>, sizeof(func2_t<double>));
    cudaMemcpyFromSymbol(&h_divide2_op, divide2_op<double>, sizeof(func2_t<double>));

    // reading file, cpu operations
    std::string h{};
    size_t nblines, nbcols;
    double *h_A = readCSV(argv[1], h, nblines, nbcols);
    auto A = CPUMatrix(h_A, nbcols, nblines);
    auto cpuSum = A.mean(0);
    auto R = A - cpuSum; // testing centered data

    // left operand
    double *d_A;
    size_t d_apitch;
    unsigned int a_0 = A.getDim0(), a_1 = A.getDim1(); //size_t width = nbcols, height = nblines;
    cudaMallocPitch(&d_A, &d_apitch, a_1 * sizeof(double), a_0 * sizeof(double));
    cudaCheckError();
    cudaMemcpy2D(d_A, d_apitch, A.getArray(), a_1 * sizeof(double), a_1 * sizeof(double), a_0, cudaMemcpyHostToDevice);
    cudaCheckError();

    // right operand
    double *d_B;
    size_t d_bpitch;
    unsigned int b_0 = cpuSum.getDim0(), b_1 = cpuSum.getDim1();
    cudaMallocPitch(&d_B, &d_bpitch, b_1 * sizeof(double), b_0 * sizeof(double));
    cudaCheckError();
    cudaMemcpy2D(d_B, d_bpitch, cpuSum.getArray(), b_1 * sizeof(double), b_1 * sizeof(double), b_0, cudaMemcpyHostToDevice);
    cudaCheckError();

    // result
    double *d_R = d_A; // in place operation
    size_t d_rpitch = d_apitch;
    unsigned int r_0, r_1;
    runtime_assert(get_broadcastable_size(a_0, a_1, b_0, b_1, &r_0, &r_1), "Invalid size for broadcasting !");
    runtime_assert(r_0 == b_0 && r_1 == b_1, "Invalid broadcasting for inplace operation !");

    // Launch the kernel
    dim3 blocksize(32,32); // 1024 threads per block TODO: change to test
    int nbblocksx = std::ceil((float)r_1 / blocksize.x);
    int nbblocksy = std::ceil((float)r_0 / blocksize.y);
    dim3 gridsize(nbblocksx, nbblocksy);
    runtime_assert(gridsize.x * gridsize.y * blocksize.x * blocksize.y >= r_0 * r_1, "Not enough threads !");
    //int threads = 4; // TODO: change this
    //int blocks = std::ceil((float)r_0 * r_1 / threads);
    //dim3 blocks(nbblocks, height);
    broadcast_op_kernel<double><<<blocksize, gridsize>>>(d_A, d_B, d_R, h_subtract2_op,
        a_0, a_1, d_apitch,
        b_0, b_1, d_bpitch,
        r_0, r_1, d_rpitch);
    cudaDeviceSynchronize();
    cudaCheckError();

    // host result
    double *h_r = (double*)malloc(r_0 * d_rpitch);
    runtime_assert(h_r != nullptr, "Alloc error !");

    // copy back to host
    cudaMemcpy(h_r, d_r, r_0 * d_rpitch, cudaMemcpyDeviceToHost);
    cudaCheckError();

    // checking result
    double *h_Rcpu = R.getArray();
    runtime_assert(r_0 == R.getDim0() && r_1 == R.getDim1(), "Invalid shapes !");
    for (size_t i = 0; i < r_0; ++i)
    {
        for (size_t j = 0; j < r_1; ++j)
        {
            if (h_r[i + j * d_rpitch] != h_Rcpu[i + j * r_1])
            {
                std::cerr << "Difference : "
                    << "GPU: " << h_r[i + j * d_rpitch]
                    << std::endl
                    << "CPU: " << h_Rcpu[i + j * r_1]
                    << std::endl;
                return EXIT_FAILURE; // Free...
            }
        }
    }

    std::cerr << "SUCCESS !" << std::endl;

    // free memory
    cudaFree(d_A);
    cudaCheckError();
    cudaFree(d_B);
    cudaCheckError();
    // in case not inplace:
    //cudaFree(d_R);
    //cudaCheckError();
    free(h_r);
}