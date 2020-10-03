#include <err.h> // errx
#include <assert.h>

#include <iostream>

#include "libalg/mean.hpp"

double *mean_axises(double *m, size_t nb_axis, size_t nb_points)
{
    size_t i, j;
    double *r;
    if (!(r = (double *)calloc(nb_axis, sizeof(double))))
        errx(2, "alloc error !");
    for (i = 0; i < nb_axis; ++i)
    {
        for (j = 0; j < nb_points; ++j)
        {
            r[i] += m[i * nb_points + j];
        }
        r[i] /= nb_points;
    }
    return r;
}