#include <cstdlib>

#include <cmath>

#include <iostream>

#include "error.hpp"
#include "libalg/alg.hpp"
#include "libalg/broadcasting.hpp"

double squared_norm_2(double a, double b)
{
    return pow((a - b), 2);
}

double add(double a, double b)
{
    return a + b;
}

double subtract(double a, double b)
{
    return a - b;
}

double mult(double a, double b)
{
    return a * b;
}

double element_wise_reduce(double *a, double *b,
                           size_t a_0, size_t a_1, size_t b_0, size_t b_1,
                           double (*op)(double, double),
                           double (*rop)(double, double),
                           double (*lop)(double, double))
{
    size_t i, j, r_0, r_1;
    double tmp;
    double r = 0;
    runtime_assert(get_broadcastable_size(a_0, a_1, b_0, b_1, &r_0, &r_1), "element_wise_reduce requires arrays to be broadcastable");
    for (i = 0; i < r_0; ++i)
    {
        tmp = 0;
        for (j = 0; j < r_1; ++j)
        {
            //std::cerr << a[(i % a_0) * a_1 + (j % a_1)] << "\t";
            tmp = rop(tmp, op(a[(i % a_0) * a_1 + (j % a_1)], b[(i % b_0) * b_1 + (j % b_1)]));
        }
        r = lop(r, tmp);
    }
    return r;
}

double *transpose(const double *a, size_t a_0, size_t a_1)
{
    size_t i, j;
    double *r;
    //if (!(r = (double *)calloc(a_0 * a_1, sizeof(double))))
    //    errx(1, "Alloc error !");
    runtime_assert((r = (double *)calloc(a_0 * a_1, sizeof(double))) != nullptr, "Alloc error !");
    for (i = 0; i < a_0; ++i)
    {
        for (j = 0; j < a_1; ++j)
        {
            r[j * a_0 + i] = a[i * a_1 + j];
        }
    }
    return r;
}

void element_wise_op(double **r, double *a, double *b,
                     size_t a_0, size_t a_1, size_t b_0, size_t b_1,
                     size_t &r_0, size_t &r_1,
                     double (*op)(double a, double b))
{
    size_t i, j, expr_0, expr_1;
    runtime_assert(get_broadcastable_size(a_0, a_1, b_0, b_1, &expr_0, &expr_1), "element_wise_op requires arrays to be broadcastable");
    if (*r == nullptr)
    {
        *r = (double *)calloc(expr_0 * expr_1, sizeof(double));
        //if (*r == nullptr)
        //    errx(1, "Alloc error !");
        runtime_assert(*r != nullptr, "Alloc error !");
    }
    else
    {
        //if (r_0 != expr_0 || r_1 != expr_1)
        //    errx(2, "Invalid size for operation !");
        runtime_assert(r_0 == expr_0 && r_1 == expr_1, "Invalid size for operation !");
    }
    r_0 = expr_0;
    r_1 = expr_1;
    for (i = 0; i < r_0; ++i)
    {
        for (j = 0; j < r_1; ++j)
        {
            *(*r + i * r_1 + j) = op(a[(i % a_0) * a_1 + (j % a_1)], b[(i % b_0) * b_1 + (j % b_1)]);
        }
    }
}

void dot_product(double **r, const double *a, const double *b,
                 size_t a_0, size_t a_1, size_t b_0, size_t b_1)
{
    size_t i, j, r_0, r_1, m;
    double tmp;
    runtime_assert(a_1 == b_0, "dot_product requires a_1 == b_0");
    r_0 = a_0;
    r_1 = b_1;
    if (*r == nullptr)
    {
        *r = (double *)calloc(r_0 * r_1, sizeof(double));
        //if (*r == nullptr)
        //    errx(1, "Alloc error !");
        runtime_assert(*r != nullptr, "Alloc error !");
    }
    for (i = 0; i < r_0; ++i)
    {
        for (j = 0; j < r_1; ++j)
        {
            tmp = 0;
            for (m = 0; m < a_1; ++m)
            {
                tmp += a[i * a_1 + m] * b[m * b_1 + j];
            }
            *(*r + i * r_1 + j) = tmp;
        }
    }
}

/**
 ** \deprecated use element_wise_op instead, but could be cool to have an optimized version
 **/
void matrix_substract_array_axises(double *a, double *b, size_t a_0, size_t a_1, size_t b_0, size_t b_1,
                                   double (*op)(double, double))
{
    size_t i, j;
    runtime_assert(a_0 == b_0 && b_1 == 1, "matrix_substract_array_axises requires a_0 == b_0 and b_1 == 1");
    for (i = 0; i < a_0; ++i)
    {
        for (j = 0; j < a_1; ++j)
        {
            a[i * a_1 + j] = op(a[i * a_1 + j], b[i]);
        }
    }
}