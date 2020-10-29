#include "libgpualg/mult.cuh"
#include <cmath>
#include "cuda_runtime.h"

#define Tile_size 2

__global__ void matrixMultiplyShared(float* A, float* B, float* C,
    int matARows, int matAColumns,
    int matBRows, int matBColumns,
    int matCRows, int matCColumns)
{
    // Tile size to store elements in shared memory
    __shared__ float sA[Tile_size][Tile_size]; 
    __shared__ float sB[Tile_size][Tile_size];

    int Row = blockDim.y * blockIdx.y + threadIdx.y; //To generate ids of threads.
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int k = 0; k < (((matAColumns - 1) / Tile_size) + 1); k++)
    {
        //Copy Data to Tile from Matrix (Global Memory to Shared Memory)
        if ((Row < matARows) && (threadIdx.x + (k * Tile_size)) < matAColumns)
        {
            sA[threadIdx.y][threadIdx.x] = A[(Row * matAColumns) + threadIdx.x + (k * Tile_size)];
        }
        else
        {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        //Copy Data to Tile from Matrix (Global Memory to Shared Memory)
        if (Col < matBColumns && (threadIdx.y + k * Tile_size) < matBRows)
        {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k * Tile_size) * matBColumns + Col];
        }
        else
        {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < Tile_size; ++j)//Multiplying Elements present in tile
        {
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    printf("%f", Cvalue);
    if (Row < matCRows && Col < matCColumns)//Saving Final result into Matrix C
    {
        C[Row * matCColumns + Col] = Cvalue;
    }
}

void matrixMultiplication(float* A, float* B, float* C, 
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns) {

    // declare the number of blocks per grid and the number of threads per block
    dim3 dimGrid((numCColumns / Tile_size) + 1, (numCRows / Tile_size) + 1, 1);//Number of Blocks required
    dim3 dimBlock(Tile_size, Tile_size, 1);//Number of threads in each block

    //@@ Launch the GPU Kernel here
    matrixMultiplyShared << <dimGrid, dimBlock >> > (A, B, C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
}