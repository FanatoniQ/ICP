#include <math.h>

#include "libalg/basic_operations.hpp"

float squared_norm_2(float a, float b)
{
    return pow((a - b), 2);
}

float add(float a, float b)
{
    return a + b;
}

float subtract(float a, float b)
{
    return a - b;
}

float mult(float a, float b)
{
    return a * b;
}

float divide(float a, float b)
{
    return a / b;
}

float pow2(float a)
{
    return pow(a, 2);
}