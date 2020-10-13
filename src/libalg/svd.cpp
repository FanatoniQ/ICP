#include <err.h>
#include <cstdlib>

#include <iostream>

#include "libalg/svd.hpp"

/* DGESVD prototype from LAPACK library */
namespace lapack
{
    extern "C"
    {
        extern void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a,
                            int *lda, double *s, double *u, int *ldu, double *vt, int *ldvt,
                            double *work, int *lwork, int *info);
    }
} // namespace lapack

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
        if ((*u = (double *)malloc(ldu * m * sizeof(double))) == nullptr)
            errx(1, "Alloc Error (u) !");
    }
    if (*sigma == nullptr) // shape: n
    {
        if ((*sigma = (double *)malloc(n * sizeof(double))) == nullptr)
            errx(1, "Alloc Error (sigma) !");
    }
    if (*vt == nullptr) // shape: n,n
    {
        if ((*vt = (double *)malloc(ldvt * n * sizeof(double))) == nullptr)
            errx(1, "Alloc Error (u) !");
    }
    lapack::dgesvd_(jobu, jobvt, &m, &n, a, &lda, *sigma, *u, &ldu, *vt, &ldvt, &wkopt, &lwork, &info);
    std::cerr << "Query DONE !" << std::endl;
    lwork = (int)wkopt;
    if ((work = (double *)malloc(lwork * sizeof(double))) == nullptr)
        errx(1, "Alloc Error (work) !");
    lapack::dgesvd_(jobu, jobvt, &m, &n, a, &lda, *sigma, *u, &ldu, *vt, &ldvt, work, &lwork, &info);
    free(work);
    if (info > 0)
    {
        errx(5, "Conversion error: %d !\n", info);
    }
}