#pragma once

/**
* Performs an SVD calculation
* Return U, S, VT in that order
* Pointers will need to be freed
* 
* Exit dimensions!
* m = row_A, n = col_A
* (row * col)
* U (m * m)
* S (n) // supposed to be m*n
* VT (n * n)
**/
void svd_gpu(double* d_A, size_t r_A, size_t c_A, double* d_U, double* d_S, double* d_VT);