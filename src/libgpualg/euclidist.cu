#include <iostream>
#include <assert.h>

#include "error.hpp"

#include "libgpualg/euclidist.cuh"
#include "libgpualg/mean.cuh"
#include "error.cuh"

__global__ void squared_norm_2_kernel(double *d_A, double *d_B, double *d_res, int pitch, int width, int height, int reducepitch)
{
    int threadid = threadIdx.x; // thread id in the block
    int lineid = blockIdx.y; // rows
    int dataid = blockIdx.x * blockDim.x + threadIdx.x; // cols
    extern __shared__ double s_data[];

    if (dataid >= width || lineid >= height)
    {
        s_data[threadid] = 0;
        return;
    }

    double *d_Aline = (double*)((char*)d_A + lineid * pitch);
    double *d_Bline = (double*)((char*)d_B + lineid * pitch);
    double tmp = d_Aline[dataid] - d_Bline[dataid];
    s_data[threadid] = tmp * tmp;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        assert(is_power_of_2(stride));
        if(threadid < stride)
            s_data[threadid] += s_data[threadid + stride];
        __syncthreads();
    }

    double *d_resline = (double*)((char*)d_res + lineid * reducepitch);
    if(threadid == 0)
        d_resline[blockIdx.x] = s_data[0];
}

__host__ double cuda_squared_norm_2(double *d_A, double *d_B, size_t width, size_t height, size_t pitch, int threads)
{
    size_t reducepitch;
    while (!is_power_of_2(threads))
        threads++;
    int nbblocksPerLine = std::ceil((float)width / threads);
    dim3 blocks(nbblocksPerLine, height);

    double *d_res;
    cudaMallocPitch(&d_res, &reducepitch, nbblocksPerLine * sizeof(double), height);
    cudaCheckError();
    cudaMemset2D(d_res, reducepitch, 0, nbblocksPerLine * sizeof(double), height);
    cudaCheckError();

    // LAUNCHING KERNEL
    std::cerr << "reducepitch: " << reducepitch << "pitch: " << pitch << std::endl;
    std::cerr << "nbthreads: " << threads << " nbblocksPerLine: " << blocks.x << " nbLines: " << blocks.y << std::endl;
    squared_norm_2_kernel<<<blocks, threads, threads * sizeof(double)>>>(d_A, d_B, d_res, pitch, width, height, reducepitch);
    cudaDeviceSynchronize();
    cudaCheckError();

    double *h_res = (double*)malloc(height * reducepitch);
    runtime_assert(h_res != nullptr, "Alloc error !");

    // COPY TO HOST
    cudaMemcpy(h_res, d_res, height * reducepitch, cudaMemcpyDeviceToHost);
    cudaCheckError();

    // FREEING DEVICE MEMORY
    cudaFree(d_res);
    cudaCheckError();

    double norm = 0;
    for (size_t i = 0; i < height; ++i)
    {
        double *h_resline = (double*)((char*)h_res + i * reducepitch);
        for (size_t j = 0; j < nbblocksPerLine; ++j)
        {
            norm += h_resline[j];
        }
    }

    return norm;
}
