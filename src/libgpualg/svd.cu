#include "libgpualg/svd.cuh"
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <tuple>
#include <cstdio>
#include <cassert>
#include <cstdlib>

void printMatrix(int m, int n, const double* A, int lda, const char* name)
{
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            double Areg = A[row + col * lda];
            printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
        }
    }
}

void svd_gpu(double* d_A, size_t r_A, size_t c_A, double *d_U, double *d_S, double *d_VT)
{
    // Error checking variables
    cusolverDnHandle_t cusolverH = NULL;

    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;

    // Dimensions
    const int m = r_A;
    const int n = c_A;
    const int lda = m;

    int* devInfo = NULL;
    double* d_work = NULL;
    double* d_rwork = NULL;

    int lwork = 0;
    int info_gpu = 0;

    // step 1: create cusolverDn/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    if (CUSOLVER_STATUS_SUCCESS != cusolver_status) {
        printf("Couldn't create the cusolver: Out of memory");
    }

    // step 2: copy A and B to device
    cudaStat3 = cudaMalloc((void**)&devInfo, sizeof(int));
    assert(cudaSuccess != cudaStat3);
    if (cudaSuccess != cudaStat3) {
        printf("Couldn't create the cusolver: Out of memory");
    }

    // step 3: query working space of SVD
    cusolver_status = cusolverDnDgesvd_bufferSize(
        cusolverH,
        m,
        n,
        &lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    if (CUSOLVER_STATUS_SUCCESS != cusolver_status) {
        printf("Couldn't query working space for the cusolver");
    }

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double) * lwork);
    assert(cudaSuccess == cudaStat1);
    if (cudaSuccess != cudaStat1) {
        printf("Couldn't create the cusolver: Out of memory");
    }

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
        lda,
        d_VT,
        lda,
        d_work,
        lwork,
        d_rwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    if (CUSOLVER_STATUS_SUCCESS != cusolver_status) {
        printf("Failed running the SVD");
    }
    assert(cudaSuccess == cudaStat1);
    if (cudaSuccess != cudaStat1) {
        printf("Failed running the SVD");
    }

    cudaStat2 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat2);
    if (cudaSuccess != cudaStat2) {
        printf("Failed copying the devInfo memory in SVD");
    }
    assert(0 == info_gpu);

    // Free ressources
    if (devInfo) cudaFree(devInfo);
    if (d_work) cudaFree(d_work);
    if (d_rwork) cudaFree(d_rwork);

    if (cusolverH) cusolverDnDestroy(cusolverH);
}
