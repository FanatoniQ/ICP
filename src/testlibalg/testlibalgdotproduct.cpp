#include <err.h>

#include <float.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <iomanip>

#include "libCSV/csv.hpp"
#include "libalg/alg.hpp"
#include "libalg/mean.hpp"

int main(int argc, char *argv[])
{
    if (argc != 3)
        errx(1, "Usage: ./testlibalgdotproduct file1 file2");
    std::cout << std::setprecision(15); //DBL_MANT_DIG);

    size_t nbaxis, nbpoints;
    std::string f1Header{}, f2Header{};
    std::ifstream file1(argv[1]);
    std::ifstream file2(argv[2]);
    double *m = readCSV(file1, f1Header, &nbaxis, &nbpoints);
    double *m_T = transpose(m, nbaxis, nbpoints);
    double *n = readCSV(file2, f2Header, &nbaxis, &nbpoints);
    //double *n_T = transpose(n, nbaxis, nbpoints);
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

    free(r);
    free(m);
    free(n);
    free(m_T);
    return EXIT_SUCCESS;
}