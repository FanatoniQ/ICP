#pragma once
#include "matrix.hpp"

/**
 * Matrix class of a single number
 */
class CPUNumber : Matrix {
    explicit CPUNumber(double *value);
    ~CPUNumber() override = default;
};

