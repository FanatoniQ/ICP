#pragma once

#include <stddef.h>

#include <iostream>

/**
 ** \brief print_matrix outputs a given matrix to the given stream
 ** 
 ** \param s the output stream
 ** \param a the matrix
 ** \param m the number of columns
 ** \param n the number of lines
 ** \param lda the number of columns per line in memory, can be != than m
 **/
void print_matrix(std::ostream &s, float *a, int m, int n, int lda);

/**
 ** \brief print_matrix outputs a given matrix to the given stream
 ** 
 ** \param s the output stream
 ** \param a the matrix
 ** \param m the number of columns
 ** \param n the number of lines
 **
 ** \note this calls the print_matrix(... lda) function with lda = m
 **/
void print_matrix(std::ostream &s, float *a, int m, int n);