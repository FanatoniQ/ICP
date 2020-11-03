#include <iostream>
#include <assert.h>

#include "error.hpp"

#include "libgpualg/euclidist.cuh"
#include "libgpualg/mean.cuh"
#include "error.cuh"

__global__ void squared_norm_2_kernel(float *d_A, float *d_B, float *d_res, int pitch, int width, int height, int reducepitch)
{
    int threadid = threadIdx.x; // thread id in the block
    int lineid = blockIdx.y; // rows
    int dataid = blockIdx.x * blockDim.x + threadIdx.x; // cols
    extern __shared__ float s_data[];

    if (dataid >= width || lineid >= height)
    {
        s_data[threadid] = 0;
        return;
    }

    float *d_Aline = (float*)((char*)d_A + lineid * pitch);
    float *d_Bline = (float*)((char*)d_B + lineid * pitch);
    float tmp = d_Bline[dataid] - d_Aline[dataid];
    s_data[threadid] = tmp * tmp;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        assert(is_power_of_2(stride));
        if(threadid < stride)
            s_data[threadid] += s_data[threadid + stride];
        __syncthreads();
    }

    float *d_resline = (float*)((char*)d_res + lineid * reducepitch);
    if(threadid == 0)
        d_resline[blockIdx.x] = s_data[0];
}

__host__ float cuda_squared_norm_2(float *d_A, float *d_B, size_t width, size_t height, size_t pitch, int threads)
{
    size_t reducepitch;
    threads = get_next_power_of_2(threads);
    int nbblocksPerLine = std::ceil((float)width / threads);
    dim3 blocks(nbblocksPerLine, height);

    float *d_res;
    cudaMallocPitch(&d_res, &reducepitch, nbblocksPerLine * sizeof(float), height);
    cudaCheckError();
    cudaMemset2D(d_res, reducepitch, 0, nbblocksPerLine * sizeof(float), height);
    cudaCheckError();

    // LAUNCHING KERNEL
    std::cerr << "reducepitch: " << reducepitch << "pitch: " << pitch << std::endl;
    std::cerr << "nbthreads: " << threads << " nbblocksPerLine: " << blocks.x << " nbLines: " << blocks.y << std::endl;
    squared_norm_2_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_A, d_B, d_res, pitch, width, height, reducepitch);
    cudaDeviceSynchronize();
    cudaCheckError();

    float *h_res = (float*)malloc(height * reducepitch);
    runtime_assert(h_res != nullptr, "Alloc error !");

    // COPY TO HOST
    cudaMemcpy(h_res, d_res, height * reducepitch, cudaMemcpyDeviceToHost);
    cudaCheckError();

    // FREEING DEVICE MEMORY
    cudaFree(d_res);
    cudaCheckError();

    float norm = 0;
    for (size_t i = 0; i < height; ++i)
    {
        float *h_resline = (float*)((char*)h_res + i * reducepitch);
        for (size_t j = 0; j < nbblocksPerLine; ++j)
        {
            norm += h_resline[j];
        }
    }

    free(h_res);
    return norm;
}

__global__ void squared_norm_2_kernel_0(const float *d_A, const float *d_B, float *d_res, int pitch, int width, int height, int reducepitch)
{
    extern __shared__ float s_data[]; // s_data is of size blockDim.y
    int threadid = threadIdx.x; // thread id in the block
    int lineid = blockIdx.y * blockDim.x + threadIdx.x; // line
    int dataid = blockIdx.x; // column
    if (dataid >= width || lineid >= height) {
        s_data[threadid] = 0; // prevent other threads from adding uninit values
        return;
    }
    // each thread copies to shared memory
    float *d_Aline = (float *)((char *)d_A + lineid * pitch);
    float *d_Bline = (float *)((char *)d_B + lineid * pitch);
    float tmp = d_Bline[dataid] - d_Aline[dataid];
    s_data[threadid] = tmp * tmp;
    printf("%d,%d,%d: %lf - %lf = %lf; ^2 = %lf \n", lineid, dataid, threadid, d_Bline[dataid], d_Aline[dataid], tmp, s_data[threadid]);
    __syncthreads();
    // each thread will reduce with one other shared data element in the middle right part of s_data
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        assert(is_power_of_2(stride)); // if not power of 2 ...
        if (threadid < stride) {// a lot of threads are idle...
             printf("%d,%d,%d: %lf + %lf = %lf\n", lineid, dataid, threadid, s_data[threadid], s_data[threadid + stride], s_data[threadid] + s_data[threadid + stride]);
             s_data[threadid] += s_data[threadid + stride];
	}
        __syncthreads();
    }
    float *d_resline = (float *)((char *)d_res + blockIdx.y * reducepitch);
    if (threadid == 0) {
        printf("%d,%d,%d = %lf\n", lineid, dataid, threadid, s_data[0]);
        d_resline[blockIdx.x] = s_data[0];
    }
}

__host__ float cuda_squared_norm_2_0(float *d_A, float *d_B, size_t width, size_t height, size_t pitch, int threads)
{
    size_t reducepitch;
    threads = get_next_power_of_2(threads);
    int nbblocksPerCol = std::ceil((float)height / threads);
    dim3 blocks(width, nbblocksPerCol);

    float *d_res;
    cudaMallocPitch(&d_res, &reducepitch, width * sizeof(float), nbblocksPerCol);
    cudaCheckError();
    cudaMemset2D(d_res, reducepitch, 0, width * sizeof(float), nbblocksPerCol);
    cudaCheckError();

    // LAUNCHING KERNEL
    std::cerr << "reducepitch: " << reducepitch << "pitch: " << pitch << std::endl;
    std::cerr << "nbthreads: " << threads << " nbcols: " << blocks.x << " nbblocksPerColumn: " << blocks.y << std::endl;
    squared_norm_2_kernel_0<<<blocks, threads, threads * sizeof(float)>>>(d_A, d_B, d_res, pitch, width, height, reducepitch);
    cudaDeviceSynchronize();
    cudaCheckError();

    float *h_res = (float*)malloc(reducepitch);
    runtime_assert(h_res != nullptr, "Alloc error !");

    // COPY TO HOST
    cudaMemcpy(h_res, d_res, reducepitch, cudaMemcpyDeviceToHost);
    cudaCheckError();

    // FREEING DEVICE MEMORY
    cudaFree(d_res);
    cudaCheckError();

    float norm = 0;
    // float norm = h_resline[0];
    for (size_t i = 0; i < width; ++i)
    {
        float *h_resline = (float*)((char*)h_res + i * reducepitch);
        norm += h_resline[0];
    }

    free(h_res);
    return norm;
}
