#include "libgpualg/mult.cuh"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cmath>

int main(int argc, char** argv)
{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int N = 16;
    int SIZE = N * N;

    // Allocate memory on the host
    //vector<float> h_A(SIZE);
    //vector<float> h_B(SIZE);
    //vector<float> h_C(SIZE);
    float* h_A = (float*)malloc(SIZE * SIZE * sizeof(float));
    float* h_B = (float*)malloc(SIZE * SIZE * sizeof(float));
    float* h_C = (float*)malloc(SIZE * SIZE * sizeof(float));

    // Initialize matrices on the host
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = 2;//sin(i);
            h_B[i * N + j] = 2;//cos(j);
        }
    }

    // Allocate memory on the device
    //dev_array<float> d_A(SIZE); // CudaMalloc
    //dev_array<float> d_B(SIZE);
    //dev_array<float> d_C(SIZE);
    float *d_A;
    float *d_B;
    float* d_C;
    cudaMalloc(&d_A, SIZE * sizeof(float));
    cudaMalloc(&d_B, SIZE * sizeof(float));
    cudaMalloc(&d_C, SIZE * sizeof(float));

    //d_A.set(&h_A[0], SIZE); // CudaMemcpy
    //d_B.set(&h_B[0], SIZE);
    cudaMemcpy(d_A, h_A, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    matrixMultiplication(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    //d_C.get(&h_C[0], SIZE);
    cudaMemcpy(h_C, d_C, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    float* cpu_C;
    cpu_C = new float[SIZE];

    // Now do the matrix multiplication on the CPU
    float sum;
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            sum = 0.f;
            for (int n = 0; n < N; n++) {
                sum += h_A[row * N + n] * h_B[n * N + col];
            }
            cpu_C[row * N + col] = sum;
        }
    }

    double err = 0;
    // Check the result and make sure it is correct
    for (int ROW = 0; ROW < N; ROW++) {
        for (int COL = 0; COL < N; COL++) {
            err += cpu_C[ROW * N + COL] - h_C[ROW * N + COL];
        }
    }
    std::cout << "Error: " << err << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return err == 0;
}