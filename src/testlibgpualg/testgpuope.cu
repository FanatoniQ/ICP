#include <stdio.h>
#include <iostream>
#include <iomanip>

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

// TODO: export this in static lib, was linking failing or invalid device pointer

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

// MAIN

int main(int argc, char **argv)
{
    runtime_assert(argc == 4, "Usage: ./testgpuope file1 meanaxis op");
    
    // retrieving functions (this part is not required if not on __host__ function)
    //func2_t<double> h_add2_op, h_subtract2_op, h_mult2_op, h_divide2_op;
    /**
    cudaMemcpyFromSymbol(&h_add2_op, add2_op<double>, sizeof(func2_t<double>));
    cudaCheckError();
    cudaMemcpyFromSymbol(&h_subtract2_op, subtract2_op<double>, sizeof(func2_t<double>));
    cudaCheckError();
    cudaMemcpyFromSymbol(&h_mult2_op, mult2_op<double>, sizeof(func2_t<double>));
    cudaCheckError();
    cudaMemcpyFromSymbol(&h_divide2_op, divide2_op<double>, sizeof(func2_t<double>));
    cudaCheckError();
    **/
    // TODO:
    func2_t<double> h_op;

    // reading file, cpu operations
    std::string h{};
    size_t nblines, nbcols;
    double *h_A = readCSV(argv[1], h, nblines, nbcols);
    std::cerr << nblines << nbcols << std::endl;
    auto A = CPUMatrix(h_A, nblines, nbcols);
    std::cerr << A << std::endl;
    int axis = std::stoi(argv[2]); 
    auto cpuMean = A.mean(axis);
    // transpose if axis is 1 
    if (axis == 1)
        cpuMean = cpuMean.transpose();

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
    unsigned int b_0 = cpuMean.getDim0(), b_1 = cpuMean.getDim1();
    cudaMallocPitch(&d_B, &d_bpitch, b_1 * sizeof(double), b_0 * sizeof(double));
    cudaCheckError();
    cudaMemcpy2D(d_B, d_bpitch, cpuMean.getArray(), b_1 * sizeof(double), b_1 * sizeof(double), b_0, cudaMemcpyHostToDevice);
    cudaCheckError();

    // result
    double *d_R = d_A; // in place operation
    size_t d_rpitch = d_apitch;
    size_t r_0, r_1;
    runtime_assert(get_broadcastable_size(a_0, a_1, b_0, b_1, &r_0, &r_1), "Invalid size for broadcasting !");
    runtime_assert(r_0 == a_0 && r_1 == a_1, "Invalid broadcasting for inplace operation !");

    // Launch the kernel
    dim3 blocksize(32,32); // 1024 threads per block TODO: change to test
    int nbblocksx = std::ceil((float)r_1 / blocksize.x);
    int nbblocksy = std::ceil((float)r_0 / blocksize.y);
    dim3 gridsize(nbblocksx, nbblocksy);
    runtime_assert(gridsize.x * gridsize.y * blocksize.x * blocksize.y >= r_0 * r_1, "Not enough threads !");
    //int threads = 4; // TODO: change this
    //int blocks = std::ceil((float)r_0 * r_1 / threads);
    //dim3 blocks(nbblocks, height);
    //broadcast_op_kernel<double><<<gridsize, blocksize>>>(d_A, d_B, d_R, h_subtract2_op,
    std::cerr << d_apitch << std::endl;
    std::cerr << d_bpitch << std::endl;
    std::cerr << b_0 << "," << b_1 << std::endl;
    //broadcast_subtract_kernel<<<gridsize, blocksize>>>(d_A, d_B, d_R,
    //auto R = A - cpuMean;
    
    //runtime_assert(R.getArray() == nullptr, "Not standard empty init CPUMatrix behaviour");
    if (strcmp(argv[3], "-") == 0)
    {
         A -= cpuMean; // testing centered data
	 cudaMemcpyFromSymbol(&h_op, subtract2_op<double>, sizeof(func2_t<double>));
         cudaCheckError();
/**broadcast_op_kernel<double><<<gridsize, blocksize>>>(d_A, d_B, d_R, h_subtract2_op,
        a_0, a_1, d_apitch / sizeof(double),
        b_0, b_1, d_bpitch / sizeof(double),
        r_0, r_1, d_rpitch / sizeof(double));**/
    }
    else if (strcmp(argv[3], "+") == 0)
    {
         A += cpuMean;
	 cudaMemcpyFromSymbol(&h_op, add2_op<double>, sizeof(func2_t<double>));
         cudaCheckError();
/**broadcast_op_kernel<double><<<gridsize, blocksize>>>(d_A, d_B, d_R, h_add2_op,
        a_0, a_1, d_apitch / sizeof(double),
        b_0, b_1, d_bpitch / sizeof(double),
        r_0, r_1, d_rpitch / sizeof(double));**/
    }
    else if (strcmp(argv[3], "x") == 0)
    {
         A *= cpuMean;
	 cudaMemcpyFromSymbol(&h_op, mult2_op<double>, sizeof(func2_t<double>));
         cudaCheckError();
/**broadcast_op_kernel<double><<<gridsize, blocksize>>>(d_A, d_B, d_R, h_mult2_op,
        a_0, a_1, d_apitch / sizeof(double),
        b_0, b_1, d_bpitch / sizeof(double),
        r_0, r_1, d_rpitch / sizeof(double));**/
    }
    else if (strcmp(argv[3], "/") == 0)
    {
         A /= cpuMean;
	 cudaMemcpyFromSymbol(&h_op, divide2_op<double>, sizeof(func2_t<double>));
         cudaCheckError();
/**broadcast_op_kernel<double><<<gridsize, blocksize>>>(d_A, d_B, d_R, h_divide2_op,
        a_0, a_1, d_apitch / sizeof(double),
        b_0, b_1, d_bpitch / sizeof(double),
        r_0, r_1, d_rpitch / sizeof(double));**/
    }
    else
    {
        std::cerr << "Invalid op" << std::endl;
        return EXIT_FAILURE;
    }

    /**broadcast_subtract_kernel<<<gridsize, blocksize>>>(d_A, d_B, d_R,
        a_0, a_1, d_apitch / sizeof(double),
        b_0, b_1, d_bpitch / sizeof(double),
        r_0, r_1, d_rpitch / sizeof(double));**/
    if (b_0 == 1 && b_1 == 1) {
        broadcast_op_scalar_kernel<double><<<gridsize, blocksize>>>(d_A, d_B, d_R, h_op,
            a_0, a_1, d_apitch / sizeof(double),
            r_0, r_1, d_rpitch / sizeof(double));
    } else if (b_0 == 1) {
        broadcast_op_line_vector_kernel<double><<<gridsize, blocksize, blocksize.x * sizeof(double)>>>(d_A, d_B, d_R, h_op,
            a_0, a_1, d_apitch / sizeof(double),
            b_0, b_1, d_bpitch / sizeof(double),
            r_0, r_1, d_rpitch / sizeof(double));
    } else if (b_1 == 1) {
        broadcast_op_column_vector_kernel<double><<<gridsize, blocksize, blocksize.x * sizeof(double)>>>(d_A, d_B, d_R, h_op,
            a_0, a_1, d_apitch / sizeof(double),
            b_0, b_1, d_bpitch / sizeof(double),
            r_0, r_1, d_rpitch / sizeof(double));
    } else {
        broadcast_op_kernel<double><<<gridsize, blocksize>>>(d_A, d_B, d_R, h_op,
            a_0, a_1, d_apitch / sizeof(double),
            b_0, b_1, d_bpitch / sizeof(double),
            r_0, r_1, d_rpitch / sizeof(double));
    }
    cudaDeviceSynchronize();
    cudaCheckError();

    std::cerr << "FINISHED !" << std::endl;

    // host result
    double *h_r = (double*)malloc(r_0 * d_rpitch);
    runtime_assert(h_r != nullptr, "Alloc error !");

    // copy back to host
    cudaMemcpy(h_r, d_R, r_0 * d_rpitch, cudaMemcpyDeviceToHost);
    cudaCheckError();

    // checking result
    std::cerr << cpuMean << std::endl;
    std::cerr << A << std::endl;
    double *h_Rcpu = A.getArray();
    runtime_assert(r_0 == A.getDim0() && r_1 == A.getDim1(), "Invalid shapes !");
    for (size_t i = 0; i < r_0; ++i)
    {
        for (size_t j = 0; j < r_1; ++j)
        {
	    std::cerr << h_r[i * (d_rpitch / sizeof(double)) + j] << " ";
            if (h_r[j + i * (d_rpitch / sizeof(double))] != h_Rcpu[j + i * r_1])
            {
                std::cerr << i << "," << j << " : Difference : "
                    << "GPU: " << h_r[j + i * (d_rpitch / sizeof(double))]
                    << std::endl
                    << "CPU: " << h_Rcpu[j + i * r_1]
                    << std::endl;
                return EXIT_FAILURE; // Free...
            }
        }
	std::cerr << std::endl;
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
