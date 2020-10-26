#include "libgpualg/mult.cuh"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <stdio.h>

//Normal CPU Matrix Multiplication
void matMultiplyOnHost(float* A, float* B, float* C, int numARows,
    int numAColumns, int numBRows, int numBColumns,
    int numCRows, int numCColumns)
{
    for (int i = 0; i < numARows; i++)
    {
        for (int j = 0; j < numAColumns; j++)
        {
            C[i * numCColumns + j] = 0.0;
            for (int k = 0; k < numCColumns; k++)
            {
                C[i * numCColumns + j] += A[i * numAColumns + k] * B[k * numBColumns + j];
            }
        }
    }
    return;
}

void print_Mat(int Row, int Col, float* Mat)
{
    for (int i = 0; i < Row * Col; i++)
    {
        printf("%f  ", *(Mat + i));

        if ((i % Col) == 0)
        {
            printf("\n");
        }
    }
}

int main(int argc, char** argv)
{
    // Perform matrix multiplication C = A*B
    int h_A_row = 1;
    int h_A_col = 3;
    int h_B_row = 3;
    int h_B_col = 3;
    int h_C_row = h_A_row;
    int h_C_col = h_B_col;

    // Allocate memory on the host
    float* h_A = (float*)malloc(h_A_row * h_A_col * sizeof(float));
    float* h_B = (float*)malloc(h_B_row * h_B_col * sizeof(float));
    float* h_C = (float*)malloc(h_C_row * h_C_col * sizeof(float));

    for (int i = 0; i < h_A_row; i++) {
        for (int j = 0; j < h_A_col; j++) {
            h_A[i * h_A_row + j] = 2;//sin(i);
        }
    }
    h_A[3] = 9;
    for (int i = 0; i < h_B_row; i++) {
        for (int j = 0; j < h_B_col; j++) {
            h_B[i * h_B_row + j] = 2;//sin(i);
        }
    }
    h_B[6] = 53;

    float *d_A;
    float *d_B;
    float* d_C;
    cudaMalloc(&d_A, h_A_row * h_A_col * sizeof(float));
    cudaMalloc(&d_B, h_B_row * h_B_col * sizeof(float));
    cudaMalloc(&d_C, h_C_row * h_C_col * sizeof(float));

    cudaMemcpy(d_A, h_A, h_A_row * h_A_col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, h_B_row * h_B_col * sizeof(float), cudaMemcpyHostToDevice);

    matrixMultiplication(d_A, d_B, d_C, h_A_row, h_A_col, h_B_row, h_B_col, h_C_row, h_C_col);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, h_C_row * h_C_col * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    float* cpu_C;
    cpu_C = new float[h_C_row * h_C_col];

    matMultiplyOnHost(h_A, h_B, cpu_C, h_A_row, h_A_col, h_B_row, h_B_col, h_C_row, h_C_col);

    double err = 0;
    // Check the result and make sure it is correct
    for (int i = 0; i < h_C_col * h_C_row; i++) {
        err += cpu_C[i] - h_C[i];
        if (cpu_C[i] != h_C[i])
        {
            printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i / h_C_col, i % h_C_col, cpu_C[i], h_C[i]);
            break;
        }
    }
    //print_Mat(h_C_row, h_C_col, h_C);
    cudaDeviceSynchronize();
    //print_Mat(h_C_row, h_C_col, cpu_C);
    std::cout << "Error: " << err << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}