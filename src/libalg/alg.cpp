#include <cstdlib>

#include <cmath>

#include <iostream>

#include "error.hpp"
#include "libalg/alg.hpp"
#include "libalg/broadcasting.hpp"
#include "libalg/basic_operations.hpp"

float element_wise_reduce(float *a, float *b,
                           size_t a_0, size_t a_1, size_t b_0, size_t b_1,
                           float (*op)(float, float),
                           float (*rop)(float, float),
                           float (*lop)(float, float))
{
    size_t i, j, r_0, r_1;
    float tmp;
    float r = 0;
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

float *transpose(const float *a, size_t a_0, size_t a_1)
{
    size_t i, j;
    float *r = (float *)calloc(a_0 * a_1, sizeof(float));
    runtime_assert(r != nullptr, "Alloc error !");
    for (i = 0; i < a_0; ++i)
    {
        for (j = 0; j < a_1; ++j)
        {
            r[j * a_0 + i] = a[i * a_1 + j];
        }
    }
    return r;
}

void element_wise_op(float **r, float *a, float *b,
                     size_t a_0, size_t a_1, size_t b_0, size_t b_1,
                     size_t &r_0, size_t &r_1,
                     float (*op)(float a, float b))
{
    size_t i, j, expr_0, expr_1;
    runtime_assert(get_broadcastable_size(a_0, a_1, b_0, b_1, &expr_0, &expr_1), "element_wise_op requires arrays to be broadcastable");
    if (*r == nullptr)
    {
        *r = (float *)calloc(expr_0 * expr_1, sizeof(float));
        runtime_assert(*r != nullptr, "Alloc error !");
    }
    else
    {
        runtime_assert(r_0 == expr_0 && r_1 == expr_1, "Invalid size for operation !");
    }
    r_0 = expr_0;
    r_1 = expr_1;
    runtime_assert(a_0 == r_0, "Operation (a_0) will Not Supported on GPU");
    runtime_assert(a_1 == r_1, "Operation (a_1) will Not Supported on GPU");
    for (i = 0; i < r_0; ++i)
    {
        for (j = 0; j < r_1; ++j)
        {
            *(*r + i * r_1 + j) = op(a[(i % a_0) * a_1 + (j % a_1)], b[(i % b_0) * b_1 + (j % b_1)]);
        }
    }
}

void dot_product(float **r, const float *a, const float *b,
                 size_t a_0, size_t a_1, size_t b_0, size_t b_1)
{
    size_t i, j, r_0, r_1, m;
    float tmp;
    runtime_assert(a_1 == b_0, "dot_product requires a_1 == b_0");
    r_0 = a_0;
    r_1 = b_1;
    if (*r == nullptr)
    {
        *r = (float *)calloc(r_0 * r_1, sizeof(float));
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
void matrix_substract_array_axises(float *a, float *b, size_t a_0, size_t a_1, size_t b_0, size_t b_1,
                                   float (*op)(float, float))
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
