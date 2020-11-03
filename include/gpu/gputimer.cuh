#pragma once

class GPUTimer
{
protected:
    cudaEvent_t start;
    cudaEvent_t stop;

public:
    // Constructor
    GPUTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    // Destructor
    ~GPUTimer();

    // Start timer
    void Start();

    // Stop timer
    void Stop();

    /**
     * Return the elapsed time between start and stop of the timer in milliseconds
     * @return float
    */
    float ElapsedTime();
};
