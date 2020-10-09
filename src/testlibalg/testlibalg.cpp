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

int main(int argc, char *argv[])
{
    if (argc != 3 && argc != 4)
        errx(1, "Usage: ./testlibalg file1 mean | ./testlibalg file1 file2 dotproduct");

    std::cout << std::setprecision(15); //DBL_MANT_DIG);

    if (argc == 3)
    {
        if (strcmp(argv[2], "mean") != 0)
            errx(1, "Usage: ./testlibalg file1 mean | ./testlibalg file1 file2 dotproduct");
        return test_mean(argv);
    }
    else if (argc == 4)
    {
        if (strcmp(argv[3], "dotproduct") != 0)
            errx(1, "Usage: ./testlibalg file1 mean | ./testlibalg file1 file2 dotproduct");
        return test_dotproduct(argv);
    }
}