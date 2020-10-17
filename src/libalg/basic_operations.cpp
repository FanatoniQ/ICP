#include <math.h>

#include "libalg/basic_operations.hpp"

double squared_norm_2(double a, double b)
{
    return pow((a - b), 2);
}

double add(double a, double b)
{
    return a + b;
}

double subtract(double a, double b)
{
    return a - b;
}

double mult(double a, double b)
{
    return a * b;
}

double divide(double a, double b)
{
    return a / b;
}