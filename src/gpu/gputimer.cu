#include "gpu/gputimer.cuh"

GPUTimer::~GPUTimer()
{
    cudaEventDestroy(GPUTimer::start);
    cudaEventDestroy(GPUTimer::stop);
}

void GPUTimer::Start()
{
    cudaEventRecord(GPUTimer::start);
}

void GPUTimer::Stop()
{
    cudaEventRecord(GPUTimer::stop);
}

float GPUTimer::ElapsedTime()
{
    float elapsedTime;
    cudaEventSynchronize(GPUTimer::stop);
    cudaEventElapsedTime(&elapsedTime, GPUTimer::start, GPUTimer::stop);
    return elapsedTime;
}
