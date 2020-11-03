#include <cstdlib>

#include <iostream>

#include "error.hpp"
#include "libalg/svd.hpp"

#define MIN(a, b) a < b ? a : b
#define MAX(a, b) a < b ? b : a

/* DGESVD prototype from LAPACK library */
namespace lapack
{
    extern "C"
    {
#define dgesvd dgesvd_
        extern void dgesvd(char *jobu, char *jobvt, int *m, int *n, float *a,
                           int *lda, float *s, float *u, int *ldu, float *vt, int *ldvt,
                           float *work, int *lwork, int *info);
    }
} // namespace lapack

float *linearize(const float *a, int n, int m, int lda)
{
    int i, j;
    auto *r = (float *)malloc(n * m * sizeof(float));
    runtime_assert(r != nullptr, "Alloc error !");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++)
            r[j + i * m] = a[j + i * lda];
    }
    return r;
}

//void svd(float *a, float **u, float **sigma, float **vt, int m, int n, int &ldvt, int &ldu);
void svd(float *a, float **u, float **sigma, float **vt, int m, int n, int *size_s)
{
    char jobu[] = {"All"}, jobvt[] = {"All"};
    // a is of size lda,n; default is lda = m
    int lda = m, ldu = m, ldvt = n;
    *size_s = MIN(m, n);
    int lwork = -1, info;
    float wkopt;
    float *work;
    // a should be n * lda shape
    runtime_assert(lda >= (MAX(1, m)), "lda >= MAX(1,M)");
    runtime_assert(ldu >= m, "ldu >= m");
    runtime_assert(ldvt >= n, "ldvt >= n");
    if (*u == nullptr) // shape: m,m
    {
        *u = (float *)malloc(ldu * m * sizeof(float));
        runtime_assert(*u != nullptr, "Alloc Error (u) !");
    }
    if (*sigma == nullptr) // shape: n
    {
        *sigma = (float *)malloc(*size_s * sizeof(float));
        runtime_assert(*sigma != nullptr, "Alloc Error (sigma) !");
    }
    if (*vt == nullptr) // shape: n,n
    {
        *vt = (float *)malloc(ldvt * n * sizeof(float));
        runtime_assert(*vt != nullptr, "Alloc Error (u) !");
    }
    lapack::dgesvd(jobu, jobvt, &m, &n, a, &lda, *sigma, *u, &ldu, *vt, &ldvt, &wkopt, &lwork, &info);
    std::cerr << "Query DONE !" << std::endl;
    lwork = (int)wkopt;
    work = (float *)malloc(lwork * sizeof(float));
    runtime_assert(work != nullptr, "Alloc Error (work) !");
    lapack::dgesvd(jobu, jobvt, &m, &n, a, &lda, *sigma, *u, &ldu, *vt, &ldvt, work, &lwork, &info);
    free(work);
    runtime_assert(info >= 0, "Conversion error !");
    std::cerr << "lds: " << lda << "," << ldu << "," << ldvt << std::endl;
}