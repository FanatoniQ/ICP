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

// explicit instanciation for lib import

template
__global__ void broadcast_op_kernel<double>(const double *d_A, double *d_B, double *d_R, func2_t<double> op,
    unsigned int a_0, unsigned int a_1, size_t d_apitch,
    unsigned int b_0, unsigned int b_1, size_t d_bpitch,
    unsigned int r_0, unsigned int r_1, size_t d_rpitch);

