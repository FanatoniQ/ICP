#pragma once
#include "matrix.hpp"

/**
 * Matrix class of a single number
 */
class CPUNumber : public Matrix
{
private:
    double value;

public:
    CPUNumber(double value);
    ~CPUNumber() override;
};
