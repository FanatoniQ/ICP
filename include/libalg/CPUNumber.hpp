#pragma once

#include "libalg/CPUMatrix.hpp"

/**
 * CPUMatrix class of a single number
 */
class CPUNumber : public CPUMatrix
{
private:
    float value;

public:
    CPUNumber(float value);
    ~CPUNumber();
};
