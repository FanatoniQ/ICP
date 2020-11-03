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
* S (n)
* VT (m * n)
**/
void svd_gpu(float* d_A, size_t r_A, size_t c_A, float* d_U, float* d_S, float* d_VT);