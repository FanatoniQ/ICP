#pragma once

#include <stddef.h>

// numpy-style broadcasting

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
bool broadcastable(size_t a_0, size_t a_1, size_t b_0, size_t b_1);

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
bool get_broadcastable_size(size_t a_0, size_t a_1, size_t b_0, size_t b_1, size_t *r_0, size_t *r_1);