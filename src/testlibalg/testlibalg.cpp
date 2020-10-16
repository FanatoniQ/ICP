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
#include "libalg/CPUMatrix.hpp"

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
    //if (n < m)
    //    SWAP(n, m);
    //assert(nbaxis < nbpoints); // just checking
    double *a_T = transpose(a, nbaxis, nbpoints);
    free(a);
    double *u = NULL, *sigma = NULL, *vt = NULL;
    svd(a_T, &u, &sigma, &vt, m, n);

    /** Full matrices exemple: **/
    //print_matrix(std::cout, vt, n, n, n); // shape is: n,n
    /** Not full matrices **/
    print_matrix(std::cout, vt, n, n, n);    // not full matrices
    print_matrix(std::cout, sigma, n, 1, 1); // shape is: n,
    print_matrix(std::cout, u, m, n, m);     // shape is: m,n (doc says m,m...)
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
    size_t Pdim0, Pdim1;
    size_t Qdim0, Qdim1;
    std::string h{};
    double *Parray = readCSV(argv[1], h, Pdim0, Pdim1);
    double *Qarray = readCSV(argv[2], h, Qdim0, Qdim1);

    auto P = CPUMatrix(Parray, Pdim0, Pdim1);
    auto Q = CPUMatrix(Qarray, Qdim0, Qdim1);
    Q = Q.transpose();

    auto R = P.dot(Q);
    print_matrix(std::cout, R.getArray(), R.getDim1(), R.getDim0());

    return EXIT_SUCCESS;

    /**
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
    return EXIT_SUCCESS; **/
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