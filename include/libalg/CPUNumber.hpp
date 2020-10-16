#pragma once

#include "libalg/CPUMatrix.hpp"

/**
 * CPUMatrix class of a single number
 */
class CPUNumber : public CPUMatrix
{
private:
    double value;

public:
    CPUNumber(double value);
    ~CPUNumber();
};
