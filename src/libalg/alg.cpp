#include <assert.h>
#include <stdlib.h>
#include <err.h>

#include "libalg/alg.hpp"

double substract(double a, double b)
{
    return a - b;
}

/**
 ** \brief broadcastable returns wether two shapes are broadcastable
 **
 ** \param a_0: number of lines in array a
 ** \param a_1: number of columns in array a
 ** \param b_0: number of lines in array b
 ** \param b_1: number of columns in array b
 **
 ** \return true if arrays are numpy-style broadcastable
 **/
bool broadcastable(size_t a_0, size_t a_1, size_t b_0, size_t b_1)
{
    if (b_1 != a_1 && b_1 != 1 && a_1 != 1)
        return false;
    if (b_0 != a_0 && b_0 != 1 && a_0 != 1)
        return false;
    return true;
}

/**
 ** \brief get_broadcastable_size returns wether two shapes are broadcastable
 ** and fills r_0 and r_1 size for resulting array if they are
 **
 ** \param a_0: number of lines in array a
 ** \param a_1: number of columns in array a
 ** \param b_0: number of lines in array b
 ** \param b_1: number of columns in array b
 ** \param r_0: number of lines in resulting array r
 ** \param r_1: number of columns in resulting array r
 **
 ** \return true if arrays are numpy-style broadcastable
 **/
bool get_broadcastable_size(size_t a_0, size_t a_1, size_t b_0, size_t b_1, size_t *r_0, size_t *r_1)
{
    if (!broadcastable(a_0, a_1, b_0, b_1))
        return false;
    *r_0 = MAX(a_0, b_0);
    *r_1 = MAX(a_1, b_1);
    return true;
}

/**
 ** \brief element_wise_op apply the given operator element wise
 ** performs numpy style broadcasting
 ** if the arrays are not broadcastable or the resulting array
 ** cannot be allocated, exit
 **
 ** \param r: the pointer to the resulting array, if NULL
 ** this function allocates resulting the broadcastable size
 ** \param a_0: number of lines in array a
 ** \param a_1: number of columns in array a
 ** \param b_0: number of lines in array b
 ** \param b_1: number of columns in array b
 ** \param op: a function pointer for performing operations
 ** with two doubles
 **/
void element_wise_op(double **r, double *a, double *b,
                     size_t a_0, size_t a_1, size_t b_0, size_t b_1,
                     double (*op)(double a, double b))
{
    size_t i, j, r_0, r_1;
    assert(get_broadcastable_size(a_0, a_1, b_0, b_1, &r_0, &r_1));
    if (*r == NULL)
    {
        *r = (double *)calloc(r_0 * r_1, sizeof(double));
        if (*r == NULL)
            errx(1, "Alloc error !");
    }
    for (i = 0; i < r_0; ++i)
    {
        for (j = 0; j < r_1; ++j)
        {
            *(*r + i * r_1 + j) = op(a[(i % a_0) * a_1 + (j % a_1)], b[(i % b_0) * b_1 + (j % b_1)]);
        }
    }
}

/**
 ** \deprecated use element_wise_op instead, but could be cool to have an optimized version
 **/
void matrix_substract_array_axises(double *a, double *b, size_t a_0, size_t a_1, size_t b_0, size_t b_1,
                                   double (*op)(double a, double b))
{
    size_t i, j;
    assert(a_0 == b_0 && b_1 == 1); // a has a_0 lines and a_1 columns
    for (i = 0; i < a_0; ++i)
    {
        for (j = 0; j < a_1; ++j)
        {
            a[i * a_1 + j] = op(a[i * a_1 + j], b[i]);
        }
    }
}