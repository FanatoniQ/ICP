#pragma once

#define is_power_of_2(x) (x & (x-1)) == 0

enum MatrixReduceOP
{
    SUM = 0,
    MEAN = 1
};

/**
 ** \brief get_next_power_of_2 returns the ceiling power of 2 from the given number
 ** function will warn use when the ceiling power of 2 is greater than 1024, loop
 ** unrolling perfomed for cases lower than 1024, will return value if already a
 ** power of 2
 **
 ** \param value the number to get the ceiling power of 2
 ** \return r the ceiling power of two
 **/
unsigned int get_next_power_of_2(unsigned int value);

/** AXIS = 1 REDUCTION **/

/**
 ** \brief dumb_sum_kernel a dummy kernel to compute axis=1 sum
 ** each thread will sum one line with coallesced access then write
 ** the sum to global device memory at address d_sumA[idx] with
 ** idx being the threadIdx.x. In order to get the sum for each lines
 ** we need to launch nblines kernels. Each thread will call the
 ** get_line_sum function.
 ** Kernel should be launched with <<<1, height>>>, height must be <= maximum
 ** number of threads per blocks or the line height.
 ** \note kernel tree_reduce_sum_kernel should be more efficient, computing partial
 ** sums in each blocks, with a cost of log2(width/2) iterations.
 **
 ** \param d_A the device pointer
 ** \param d_sumA the device sum pointer used to store sum results
 ** \param pitch the pitch in bytes
 ** \param width the number of values in a line
 ** \param height the number of lines
 **/
 __global__ void dumb_sum_kernel(const char *d_A, float *d_sumA, int pitch, int width, int height);

 /**
 ** \brief dumb_mean_kernel a dummy kernel to compute axis=1 mean
 ** This kernel is basically the same as the dumb_sum_kernel, but
 ** each thread calls the get_line_mean function instead of get_line_sum.
 ** In order to get the mean for each lines we need to launch nblines kernels.
 **
 ** \param d_A the device pointer
 ** \param d_meanA the device mean pointer used to store mean results
 ** \param pitch the pitch in bytes
 ** \param width the number of values in a line
 ** \param height the number of lines
 **/
__global__ void dumb_mean_kernel(const char *d_A, float *d_meanA, int pitch, int width, int height);

/**
 ** \brief tree_reduce_sum_kernel computes axis=1 sum, each block of threads handles a partial
 ** line sum, should be called with <<<dim3(x,height),nbthreads, nbthreads>>> with x being the number of
 ** blocks per line. Shared memory of size nbthreads is used to store device mem d_A elements.
 ** Each thread will sum the data in the shared memory data array at index threadIdx.x with the
 ** shared memory content at index threadIdx.x + stride. At each iterations, we wait for all threads to
 ** compute the sum, then we decrease the stride, until it reaches 0. Some threads are therefore inactive
 ** but the overall number of cycles (ignoring sum cost) is log2(nbthreads/2), since stride starts at
 ** nbthreads / 2.
 **
 ** \param d_A the device pointer
 ** \param d_sumA the device mean pointer used to store mean results (each column is a partial sum
 ** and each row contains partial sums for the given d_A line
 ** \param pitch the pitch of d_A in bytes
 ** \param width the number of values in a line
 ** \param height the number of lines
 ** \param reducepitch the pitch of d_sumA array in bytes
 **/
__global__ void tree_reduce_sum_kernel(const float *d_A, float *d_sumA, int pitch, int width, int height, int reducepitch);

/**
 ** \brief tree_reduce_mean_kernel is the same as tree_reduce_mean_kernel except than we divide the last block sum
 ** by the width of the d_A matrix. Be sure to call this function on last reduce iteration.
 ** \see tree_reduce_sum_kernel
 **/
__global__ void tree_reduce_mean_kernel(const float *d_A, float *d_sumA, int pitch, int width, int height, int reducepitch);

/** AXIS = 0 REDUCTION **/

/**
 ** \brief dumb_sum_kernel_0 is the same as dumb_sum_kernel but for axis=0 instead of 1
 ** Kernel should be launched with <<<1, width>>>, width must be at most maximum
 ** number of threads per blocks and can be the matrix height. Each block will handle a column sum
 ** therefore we have width blocks. Access are not coalesced since column sum, but should be better than
 ** atomicadd.
 ** \see dumb_sum_kernel
 **
 ** \param d_A device pointer to matrix A
 ** \param d_sum device pointer for result sum line vector
 ** \param pitch d_A pitch NOT in bytes
 ** \param width the number of columns in d_A
 ** \param height the number of lines in d_A
 **/
__global__ void dumb_sum_kernel_0(const float *d_A, float *d_sumA, int pitch, int width, int height);

/**
 ** \brief dumb_mean_kernel_0 is the same as dumb_sum_kernel_0 but for mean computation
 ** \see dumb_sum_kernel_0
 **/
__global__ void dumb_sum_kernel_0(const float *d_A, float *d_meanA, int pitch, int width, int height);

/**
 ** \brief tree_reduce_sum_kernel_0 computes axis=1 sum, each block of threads handles a partial
 ** column sum, should be called with <<<dim3(width,y),nbthreads, nbthreads>>> with y being the number of
 ** blocks per column. Shared memory of size nbthreads is used to store device mem d_A elements.
 ** Each thread will sum the data in the shared memory data array at index threadIdx.x with the
 ** shared memory content at index threadIdx.x + stride.
 ** Global memory accesses in each blocks are not coallesced since we compute column sum, however, this is
 ** faster than using global atomics,
 ** At each iterations, we wait for all threads to compute the sum,
 ** then we decrease the stride, until it reaches 0. Some threads are therefore inactive
 ** but the overall number of cycles (ignoring sum cost) is log2(nbthreads/2), since stride starts at
 ** nbthreads / 2.
 ** For coalesced access we can transpose and call tree_reduce_sum_kernel
 ** \see tree_reduce_sum_kernel
 ** \todo TODO: Benchmark
 **
 ** \param d_A the device pointer
 ** \param d_sumA the device mean pointer used to store mean results, each column contains at each line,
 ** the partial sum of the column for the d_A pointer
 ** \param pitch the pitch of d_A in bytes
 ** \param width the number of values in a line
 ** \param height the number of lines
 ** \param reducepitch the pitch of d_sumA array in bytes
 **/
__global__ void tree_reduce_sum_kernel_0(const float *d_A, float *d_sumA, int pitch, int width, int height, int reducepitch);

/**
 ** \brief tree_reduce_mean_kernel_0 is the same as tree_reduce_sum_kernel_0 except than we divide the last block sum
 ** by the height of the d_A matrix. Be sure to call this function on last reduce iteration. denom is the denominator (total height)
 ** \see tree_reduce_sum_kernel_0
 **/
__global__ void tree_reduce_mean_kernel_0(const float *d_A, float *d_sumA, int pitch, int width, int height, int reducepitch, unsigned int denom);

/**
 ** \brief reduce_0 wrapper around tree_reduce_sum_kernel_0 and tree_reduce_mean_kernel_0 kernel functions
 ** calling this host function will compute the sum or mean over axis = 0 for the given d_A matrix.
 ** \todo check if this works
 **
 ** \param op the MatrixReduceOP member (either MEAN or SUM)
 ** \param d_A the device matrix pointer
 ** \param d_sum a pointer to the resulting mean line vector, or nullptr (in that case this function performs allocation)
 ** WARNING: d_sum ELEMENTS SHOULD BE ZERO INITIALIZED IF ALREADY ALLOCATED
 ** \param width the width of d_A
 ** \param height the height of d_A
 ** \param pitch the pitch of d_A IN bytes
 ** \param reducepitch a pointer to the reduce d_sum matrix pitch (IN bytes)
 ** \param threads the minimal number of threads to be used in a block
 **/
__host__ void reduce_0(enum MatrixReduceOP op, float *d_A, float **d_sum, size_t width, size_t height, size_t pitch, size_t *reducepitch, int threads);


/**
 ** \deprecated use reduce_0 instead
 **/
__host__ void mean_0(float *d_A, float **d_sum, size_t width, size_t height, size_t pitch, size_t *reducepitch, int threads);
