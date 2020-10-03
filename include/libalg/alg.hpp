#pragma once

#include <stddef.h>

#define MAX(a, b) (a > b) ? a : b

// numpy-style broadcasting
bool broadcastable(size_t a_0, size_t a_1, size_t b_0, size_t b_1);

bool get_broadcastable_size(size_t a_0, size_t a_1, size_t b_0, size_t b_1, size_t *r_0, size_t *r_1);

// operations
double substract(double a, double b);

void element_wise_op(double **r, double *a, double *b,
                     size_t a_0, size_t a_1, size_t b_0, size_t b_1,
                     double (*op)(double a, double b));