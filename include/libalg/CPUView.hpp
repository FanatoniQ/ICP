#pragma once

#include "libalg/CPUMatrix.hpp"

/**
 * CPUMatrix class of a view
 */
class CPUView : public CPUMatrix
{
public:
    CPUView(double *array, size_t dim0, size_t dim1) = delete;
    CPUView(size_t dim0, size_t dim1) = delete;
    CPUView(double *array, size_t dim1);
    ~CPUView();
};
