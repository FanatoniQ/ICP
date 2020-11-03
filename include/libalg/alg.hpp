#pragma once

#include <stddef.h>

#define MAX(a, b) (a > b) ? a : b
#define MIN(a, b) (a < b) ? a : b

#define SWAP(a, b) \
    {              \
        int c = a; \
        a = b;     \
        b = c;     \
    }

// basic operations
float subtract(float a, float b);

float add(float a, float b);

float squared_norm_2(float a, float b);

float mult(float a, float b);

// advanced array operations

/**
 ** \brief transpose returns the transposed given array
 **
 ** \param a: the array of shape a_0,a_1
 ** \param a_0: number of lines in array a
 ** \param a_1: number of columns in array a
 **
 ** \return r: the transposed a array of shape a_1,a_0
 **/
float *transpose(const float *a, size_t a_0, size_t a_1);

/**
 ** \brief element_wise_reduce apply the given operator element wise
 ** and performs numpy style broadcasting, returns the reduced value
 **
 ** \param a_0: number of lines in array a
 ** \param a_1: number of columns in array a
 ** \param b_0: number of lines in array b
 ** \param b_1: number of columns in array b
 ** \param op: a function pointer for performing operations
 ** with two floats
 ** \param rop: a function pointer for performing operations
 ** with two floats, used for reduction operation
 ** \param lop: a function pointer for performing operations
 ** with two floats, used for last operation
 **
 ** \example calling with two arrays and
 ** - op being squared_norm_2
 ** - rop being add
 ** - lop being add
 ** we get squared norm 2 |.|_2 as result
 **
 ** \quid is this a good idea ?
 **/
float element_wise_reduce(float *a, float *b,
                           size_t a_0, size_t a_1, size_t b_0, size_t b_1,
                           float (*op)(float a, float b),
                           float (*rop)(float a, float b),
                           float (*lop)(float a, float b));

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
 ** \param r_0: number of lines in resulting array r
 ** \param r_1: number of columns in resulting array r
 ** \param op: a function pointer for performing operations
 ** with two floats
 **/
void element_wise_op(float **r, float *a, float *b,
                     size_t a_0, size_t a_1, size_t b_0, size_t b_1,
                     size_t &r_0, size_t &r_1,
                     float (*op)(float a, float b));

/**
 ** \brief dot_product returns the dot product of the given
 ** a and b arrays, a_1 must be equal to b_0, or exit
 **
 ** \param r: the pointer to the resulting array, if NULL
 ** this function allocates a resulting size array
 ** \param a_0: number of lines in array a
 ** \param a_1: number of columns in array a
 ** \param b_0: number of lines in array b
 ** \param b_1: number of columns in array b
 **/
void dot_product(float **r, const float *a, const float *b,
                 size_t a_0, size_t a_1, size_t b_0, size_t b_1);