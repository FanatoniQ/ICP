#pragma once
#include "matrix.hpp"

/**
 * Matrix class of a normal matrix
 */
class CPUMat : public Matrix
{
public:
    /**
    * Initialize matrix of dim0*dim1 with 0s
    * Throws std::bad_alloc on failed malloc
    * @param dim0
    * @param dim1
    */
    CPUMat(size_t dim0, size_t dim1);
    CPUMat(double *array, size_t dim0, size_t dim1);
};