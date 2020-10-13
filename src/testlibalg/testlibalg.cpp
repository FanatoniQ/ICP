#include <err.h>
#include <cassert>

#include <float.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>

#include "libCSV/csv.hpp"
#include "libalg/print.hpp"
#include "libalg/svd.hpp"
#include "libalg/alg.hpp"
#include "libalg/mean.hpp"

#define UNUSED(x) (void)x

int test_svd(char *argv[])
{
    /**
    UNUSED(argv);
    double a[] = {1.0, 2.0, 3.0};
    double b[] = {4.0, 5.0, 6.0};
    double c[] = {7.0, 8.0, 9.0};
    // TODO: use lapack for SVD computation
    print_matrix(std::cout, a, 3, 1);
    print_matrix(std::cout, b, 3, 1);
    print_matrix(std::cout, c, 3, 1);
    **/
    size_t nbaxis, nbpoints;
    std::string f1Header{};
    std::ifstream file1(argv[1]);

    double *a = readCSV(file1, f1Header, &nbaxis, &nbpoints);
    //int n = MAX(nbaxis, nbpoints), m = MIN(nbaxis, nbpoints);
    int n = nbpoints, m = nbaxis;
    if (n < m)
        SWAP(n, m);
    //assert(nbaxis < nbpoints); // just checking
    double *a_T = transpose(a, nbaxis, nbpoints);
    free(a);
    double *u = NULL, *sigma = NULL, *vt = NULL;
    svd(a_T, &u, &sigma, &vt, m, n);

    /** Full matrices exemple: **/
    print_matrix(std::cout, u, m, m, m);     // shape is: m,n (doc says m,m...)
    print_matrix(std::cout, sigma, m, 1, m); // shape is: n,
    print_matrix(std::cout, vt, n, n, n);    // shape is: n,n
    /**
    print_matrix(std::cout, u, m, m, m);
    print_matrix(std::cout, sigma, m, 1, m);
    print_matrix(std::cout, vt, n, m, n); // not full matrices
    **/
    free(u);
    free(sigma);
    free(vt);
    free(a_T);
    return EXIT_SUCCESS;
}

int test_mean(char *argv[])
{
    size_t nbaxis, nbpoints;
    std::string f1Header{};
    std::ifstream file1(argv[1]);

    double *m = readCSV(file1, f1Header, &nbaxis, &nbpoints);
    double *mean = mean_axises(m, nbaxis, nbpoints);
    std::cerr << "nbaxis: " << nbaxis << " nbpoints: " << nbpoints << std::endl;
    std::cerr << "Mean:" << std::endl;
    print_matrix(std::cout, mean, nbaxis, 1);
    free(m);
    free(mean);
    return EXIT_SUCCESS;
}

int test_dotproduct(char *argv[])
{
    size_t nbaxis, nbpoints;
    std::string f1Header{}, f2Header{};
    std::ifstream file1(argv[1]);
    std::ifstream file2(argv[2]);
    double *m = readCSV(file1, f1Header, &nbaxis, &nbpoints);
    double *m_T = transpose(m, nbaxis, nbpoints);
    double *n = readCSV(file2, f2Header, &nbaxis, &nbpoints);
    double *r = NULL;
    dot_product(&r, m_T, n, nbpoints, nbaxis, nbaxis, nbpoints);
    print_matrix(std::cout, r, nbpoints, nbpoints);
    free(m);
    free(r);
    free(n);
    free(m_T);
    return EXIT_SUCCESS;
}

void usage(void)
{
    std::cerr << "Usage:" << std::endl
              << std::endl;
    std::cerr << "./testlibalg file1 1paramfunction" << std::endl;
    std::cerr << "./testlibalg file1 file2 2paramsfunction" << std::endl
              << std::endl;
    std::cerr << "1paramfunction = mean | svd" << std::endl;
    std::cerr << "2paramsfunction = dotproduct" << std::endl;
    exit(1);
}

int main(int argc, char *argv[])
{
    std::cout << std::setprecision(15); //DBL_MANT_DIG);
    if (argc == 3)
    {
        if (strcmp(argv[2], "mean") == 0)
            return test_mean(argv);
        else if (strcmp(argv[2], "svd") == 0)
            return test_svd(argv);
    }
    else if (argc == 4)
    {
        if (strcmp(argv[3], "dotproduct") == 0)
            return test_dotproduct(argv);
    }
    usage();
}