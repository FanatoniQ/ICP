#include <err.h> // errx
#include <assert.h>

#include <math.h>

#include <iostream>

#include "libalg/mean.hpp"

#define UNUSED(x) (void)x

size_t get_id_reduce_first_axis(size_t i, size_t j, size_t dim0, size_t dim1)
{
    UNUSED(i);
    UNUSED(dim0);
    UNUSED(dim1);
    return j;
    //return (size_t)(idx % dim1); // return j
}

size_t get_id_reduce_second_axis(size_t i, size_t j, size_t dim0, size_t dim1)
{
    UNUSED(j);
    UNUSED(dim1);
    UNUSED(dim0);
    return i;
    //return (size_t)floor(idx / dim1); // return i
}

size_t get_id_reduce_flattened(size_t i, size_t j, size_t dim0, size_t dim1)
{
    UNUSED(i);
    UNUSED(j);
    UNUSED(dim0);
    UNUSED(dim1);
    return 0;
}

double *mean_axises(double *m, size_t dim0, size_t dim1, int axis)
{
    double *r = nullptr;
    size_t dimr;
    mean_axises(&r, m, dim0, dim1, dimr, axis);
    return r;
}

void sum_axises(double **r, double *m, size_t dim0, size_t dim1, size_t &dimr, int axis)
{
    size_t i, j;
    size_t (*get_id_reduce)(size_t i, size_t j, size_t dim0, size_t dim1) = NULL;
    if (axis < 0)
    {
        dimr = 1; // r[0] == r[e % 1]
        get_id_reduce = get_id_reduce_flattened;
    }
    else if (axis == 0)
    {
        dimr = dim1; // r[j] == r[e % dim0]
        get_id_reduce = get_id_reduce_first_axis;
    }
    else if (axis == 1)
    {
        dimr = dim0; // r[i] == r[e % dim1]
        get_id_reduce = get_id_reduce_second_axis;
    }
    else
        errx(3, "Invalid dimension !");
    if (*r == nullptr)
        if ((*r = (double *)calloc(dimr, sizeof(double))) == nullptr)
            errx(2, "alloc error !");
    for (i = 0; i < dim0; ++i)
    {
        for (j = 0; j < dim1; ++j)
        {
            (*r)[get_id_reduce(i, j, dim0, dim1)] += m[i * dim1 + j];
        }
    }
}

void mean_axises(double **r, double *m, size_t dim0, size_t dim1, size_t &dimr, int axis)
{
    sum_axises(r, m, dim0, dim1, dimr, axis);
    size_t i;
    size_t denom;
    if (axis < 0)
        denom = dim0 * dim1;
    else if (axis == 0)
        denom = dim0;
    else if (axis == 1)
        denom = dim1;
    else
        errx(3, "Invalid dimension !");
    for (i = 0; i < dimr; ++i)
    {
        (*r)[i] /= denom;
    }
}