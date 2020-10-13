#include <err.h>

#include <float.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>

#include "libCSV/csv.hpp"
#include "libalg/alg.hpp"
#include "libalg/mean.hpp"

#define UNUSED(x) (void)x

int test_svd(char *argv[])
{
    UNUSED(argv);
    // TODO: use lapack for SVD computation
    std::cout << "1,2,3" << std::endl
              << std::endl;
    std::cout << "4,5,6" << std::endl
              << std::endl;
    std::cout << "7,8,9" << std::endl
              << std::endl;
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
    for (size_t i = 0; i < nbaxis; ++i)
    {
        std::cout << mean[i];
        if (i != nbaxis - 1)
            std::cout << ',';
    }
    std::cout << std::endl;

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

    for (size_t i = 0; i < nbpoints; ++i)
    {
        for (size_t j = 0; j < nbpoints; ++j)
        {
            std::cout << r[i * nbpoints + j];
            if (j != nbpoints - 1)
                std::cout << ',';
        }
        std::cout << std::endl;
    }

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