#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

/**
 ** \brief readCSV returns the transposed csv file, (each line is constituted of nbaxis points)
 **
 ** \deprecated reads an input stream using c++ vector then returns the C transposed array
 **
 ** \param f csv input stream
 ** \param h the header string or empty string if no header
 ** \param nbaxis a pointer to be filled with the number of axis (aka lines once transposed)
 ** \param nbpoints a pointer to be filled with the number of lines (aka columns once transposed)
 **
 ** \return res the parsed converted to C-style array transposed vector of vector
 **/
double *readCSV(std::istream &f, std::string &h, size_t *nbaxis, size_t *nbpoints);

/**
 ** \brief readCSV C method to return the non transposed double array parsed from
 ** the given CSV file
 **
 ** \param path csv file to be opened for reading
 ** \param h the header string or empty string if no header
 ** \param nblines the number of lines in the array
 ** \param nbcols the number of columns in the array
 **
 ** \return r the (nblines x nbcols) shaped array
 **/
double *readCSV(const char *path, std::string &h, size_t &nblines, size_t &nbcols);