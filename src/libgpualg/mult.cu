#include "libgpualg/mult.cuh"
#include <cmath>
#include "cuda_runtime.h"

#define Tile_size 2

__global__ void matrixMultiplyShared(float* A, float* B, float* C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns)
{
    __shared__ float sA[Tile_size][Tile_size];   // Tile size to store elements in shared memory
    __shared__ float sB[Tile_size][Tile_size];

    int Row = blockDim.y * blockIdx.y + threadIdx.y; //To generate ids of threads.
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int k = 0; k < (((numAColumns - 1) / Tile_size) + 1); k++)
    {
        if ((Row < numARows) && (threadIdx.x + (k * Tile_size)) < numAColumns)//Copy Data to Tile from Matrix (Global Memory to Shared Memory)
        {
            sA[threadIdx.y][threadIdx.x] = A[(Row * numAColumns) + threadIdx.x + (k * Tile_size)];
        }
        else
        {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (Col < numBColumns && (threadIdx.y + k * Tile_size) < numBRows)//Copy Data to Tile from Matrix (Global Memory to Shared Memory)
        {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k * Tile_size) * numBColumns + Col];
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
    if (Row < numCRows && Col < numCColumns)//Saving Final result into Matrix C
    {
        C[Row * numCColumns + Col] = Cvalue;
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