#pragma once

#include <stdlib.h>

#include "libalg/alg.hpp"

void sum_axises(double **r, double *m, size_t dim0, size_t dim1, size_t &dimr, int axis = 0);

void mean_axises(double **r, double *m, size_t dim0, size_t dim1, size_t &dimr, int axis = 0);

/**
 ** \brief mean_axises returns the mean for the first or second axis or entire array,
 ** linear access with use of function pointers, no modulus rather param ignoring (faster)
 ** \deprecated use above function instead
 ** \todo we should define the sum function and make the above function call the sum function
 ** which is != than the element_wise_op sum, since we sum elements over axises, not two matrices
 **
 ** \param m the matrix to get the mean from
 ** \param dim0 the number of lines
 ** \param dim1 the number of columns
 ** \param axis wether to return the first axis mean or second axis mean
 ** (-1 value is for flattened mean)
 **
 ** \return r the mean over the asked axis, for example:
 ** [0 1 2
 **  1 2 0
 **  1 1 1
 **  2 2 0]
 ** axis == -1: [1.083]
 ** axis == 0: [1 1.5 0.75]
 ** axis == 1: [1 1 1 1.333]
 **/
double *mean_axises(double *m, size_t dim0, size_t dim1, int axis = 0);