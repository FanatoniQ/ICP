#pragma once

/**
* Performs an SVD calculation
* Return U, S, VT in that order
* Pointers will need to be freed
**/
std::tuple<double*, double*, double*> svd(double* A, size_t r_A, size_t c_A);