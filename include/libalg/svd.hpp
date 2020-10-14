#pragma once

/**
 ** \brief svd computes the singular value decomposition using lapack lib
 ** a = u * sigma * vt where vt is transpose(v)
 ** taken from lapack DGESVD documentation:
 ** http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html
 ** 
 ** \note u, sigma and vt are allocated if null pointers
 **
 ** \param a the input matrix (m,n) shape
 ** \param u the orthogonal (m,m) matrix, left singular vector of a
 ** \param sigma the singular values of a (real non negative, descending order)
 ** \param vt the, transposed right singular vector of a
 ** \param m the number of columns in a
 ** \param n the number of lines in a 
 **/
void svd(double *a, double **u, double **sigma, double **vt, int m, int n);