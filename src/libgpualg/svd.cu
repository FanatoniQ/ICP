#include "libgpualg/svd.cuh"
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <tuple>
#include <cstdio>
#include <cassert>
#include <cstdlib>

void printMatrix(int m, int n, const float* A, int lda, const char* name)
{
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            float Areg = A[row + col * lda];
            printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
        }
    }
}

void svd_gpu(float* d_A, size_t r_A, size_t c_A, float *d_U, float *d_S, float *d_VT)
{
    // Error checking variables
    cusolverDnHandle_t cusolverH = NULL;

    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    cudaError_t cudaStat6 = cudaSuccess;

    // Dimensions
    const int m = r_A;
    const int n = c_A;
    const int lda = m;
    /*
    // Return arrays
    float* U = (float*)malloc(lda * m * sizeof(float));
    if (U == nullptr)
        throw std::bad_alloc();
    float* VT = (float*)malloc(lda * n * sizeof(float));
    if (VT == nullptr)
        throw std::bad_alloc();
    float* S = (float*)malloc(n * sizeof(float));
    if (S == nullptr)
        throw std::bad_alloc();
    //float U[lda * m]; // m-by-m unitary matrix 
    //float VT[lda * n];  // n-by-n unitary matrix
    //float S[n]; // singular value
    */

    //float* d_A = NULL;
    //float* d_S = NULL;
    //float* d_U = NULL;
    //float* d_VT = NULL;
    int* devInfo = NULL;
    float* d_work = NULL;
    float* d_rwork = NULL;
    //float* d_W = NULL;  // W = S*VT

    int lwork = 0;
    int info_gpu = 0;
    /*
    printf("A = (matlab base-1)\n");
    printMatrix(m, n, A, lda, "A");
    printf("=====\n");
    */
    // step 1: create cusolverDn/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    // step 2: copy A and B to device
    //cudaStat1 = cudaMalloc((void**)&d_A, sizeof(float) * lda * n);
    //cudaStat2 = cudaMalloc((void**)&d_S, sizeof(float) * n);
    //cudaStat3 = cudaMalloc((void**)&d_U, sizeof(float) * lda * m);
    //cudaStat4 = cudaMalloc((void**)&d_VT, sizeof(float) * lda * n);
    cudaStat5 = cudaMalloc((void**)&devInfo, sizeof(int));

    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);
    assert(cudaSuccess == cudaStat6);

    //cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) * lda * n, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);

    // step 3: query working space of SVD
    cusolver_status = cusolverDnDgesvd_bufferSize(
        cusolverH,
        m,
        n,
        &lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float) * lwork);
    assert(cudaSuccess == cudaStat1);

    // step 4: compute SVD
    signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT
    cusolver_status = cusolverDnDgesvd(
        cusolverH,
        jobu,
        jobvt,
        m,
        n,
        d_A,
        lda,
        d_S,
        d_U,
        lda,  // ldu
        d_VT,
        lda, // ldvt,
        d_work,
        lwork,
        d_rwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);


    //cudaStat1 = cudaMemcpy(U, d_U, sizeof(float) * lda * m, cudaMemcpyDeviceToHost);
    //cudaStat2 = cudaMemcpy(VT, d_VT, sizeof(float) * lda * n, cudaMemcpyDeviceToHost);
    //cudaStat3 = cudaMemcpy(S, d_S, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cudaStat4 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    printf("after gesvd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);
    printf("=====\n");

    /*
    printf("S = (matlab base-1)\n");
    printMatrix(n, 1, S, lda, "S");
    printf("=====\n");

    printf("U = (matlab base-1)\n");
    printMatrix(m, m, U, lda, "U");
    printf("=====\n");

    printf("VT = (matlab base-1)\n");
    printMatrix(n, n, VT, lda, "VT");
    printf("=====\n");
    */
    // free resources
    //if (d_A) cudaFree(d_A);
    //if (d_S) cudaFree(d_S);
    //if (d_U) cudaFree(d_U);
    //if (d_VT) cudaFree(d_VT);
    if (devInfo) cudaFree(devInfo);
    if (d_work) cudaFree(d_work);
    if (d_rwork) cudaFree(d_rwork);
    //if (d_W) cudaFree(d_W);

    if (cusolverH) cusolverDnDestroy(cusolverH);

    //cudaDeviceReset();
    //return { U, S, VT };
}
