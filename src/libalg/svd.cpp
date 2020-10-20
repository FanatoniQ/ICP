#include <cstdlib>

#include <iostream>

#include "error.hpp"
#include "libalg/svd.hpp"

/* DGESVD prototype from LAPACK library */
namespace lapack
{
    extern "C"
    {
#define dgesvd dgesvd_
        extern void dgesvd(char *jobu, char *jobvt, int *m, int *n, double *a,
                           int *lda, double *s, double *u, int *ldu, double *vt, int *ldvt,
                           double *work, int *lwork, int *info);
    }
} // namespace lapack

double *linearize(double *a, int n, int m, int lda)
{
    int i, j;
    double *r = (double *)malloc(n * m * sizeof(double));
    runtime_assert(r != nullptr, "Alloc error !");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++)
            r[j + i * m] = a[j + i * lda];
    }
    return r;
}

void svd(double *a, double **u, double **sigma, double **vt, int m, int n)
{
    char jobu[] = {"All"}, jobvt[] = {"All"};
    // a is of size lda,n; default is lda = m
    int ldu = m, ldvt = n, lda = m;
    int lwork = -1, info;
    double wkopt;
    double *work;
    if (*u == nullptr) // shape: m,m
    {
        *u = (double *)malloc(ldu * m * sizeof(double));
        runtime_assert(*u != nullptr, "Alloc Error (u) !");
    }
    if (*sigma == nullptr) // shape: n
    {
        *sigma = (double *)malloc(n * sizeof(double));
        runtime_assert(*sigma != nullptr, "Alloc Error (sigma) !");
    }
    if (*vt == nullptr) // shape: n,n
    {
        *vt = (double *)malloc(ldvt * n * sizeof(double));
        runtime_assert(*vt != nullptr, "Alloc Error (u) !");
    }
    lapack::dgesvd(jobu, jobvt, &m, &n, a, &lda, *sigma, *u, &ldu, *vt, &ldvt, &wkopt, &lwork, &info);
    std::cerr << "Query DONE !" << std::endl;
    lwork = (int)wkopt;
    work = (double *)malloc(lwork * sizeof(double));
    runtime_assert(work != nullptr, "Alloc Error (work) !");
    lapack::dgesvd(jobu, jobvt, &m, &n, a, &lda, *sigma, *u, &ldu, *vt, &ldvt, work, &lwork, &info);
    free(work);
    runtime_assert(info <= 0, "Conversion error !");
}