#pragma once

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

/**
 ** \brief dumb_sum_kernel a dummy kernel to compute axis=1 sum
 ** each thread will sum one line with coallesced access then write
 ** the sum to global device memory at address d_sumA[idx] with
 ** idx being the threadIdx.x. In order to get the sum for each lines
 ** we need to launch nblines kernels. Each thread will call the
 ** get_line_sum function.
 ** Kernel should be launched with <<<height, width>>>, width can be maximum
 ** number of threads per blocks or the line width.
 ** \note kernel tree_reduce_sum_kernel should be more efficient, computing partial
 ** sums in each blocks, with a cost of log2(width/2) iterations.
 **
 ** \param d_A the device pointer
 ** \param d_sumA the device sum pointer used to store sum results
 ** \param pitch the pitch in bytes
 ** \param width the number of values in a line
 ** \param height the number of lines
 **/
 __global__ void dumb_sum_kernel(const char *d_A, double *d_sumA, int pitch, int width, int height);

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
__global__ void dumb_mean_kernel(const char *d_A, double *d_meanA, int pitch, int width, int height);

/**
 ** \brief tree_reduce_sum_kernel computes axis=1 mean, each block of threads handles a partial
 ** line sum, should be called with <<<dim3(x,height),nbthreads, nbthreads>>> with x being the number of
 ** blocks per line. Shared memory of size nbthreads is used to store device mem d_A elements.
 ** Each thread will sum the data in the shared memory data array at index threadIdx.x with the
 ** shared memory content at index threadIdx.x + stride. At each iterations, we wait for all threads to
 ** compute the sum, then we decrease the stride, until it reaches 0. Some threads are therefore inactive
 ** but the overall number of cycles (ignoring sum cost) is log2(nbthreads/2), since stride starts at
 ** nbthreads / 2.
 **
 ** \param d_A the device pointer
 ** \param d_meanA the device mean pointer used to store mean results (each column is a partial mean
 ** and each row contains partial means for the given d_A line
 ** \param pitch the pitch of d_A in bytes
 ** \param width the number of values in a line
 ** \param height the number of lines
 ** \param reducepitch the pitch of d_sumA array in bytes
 **/
__global__ void tree_reduce_sum_kernel(const double *d_A, double *d_sumA, int pitch, int width, int height, int reducepitch);
