#include <err.h> // errx

#include <float.h> // DBL_MANT_DIG: number of bits in mantisse
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <iomanip>

#include "libCSV/csv.hpp"
#include "libalg/mean.hpp"

int main(int argc, char *argv[])
{
    if (argc != 3)
        errx(1, "Usage: ./CPUICP file1 file2");
    std::cout << std::setprecision(15); //DBL_MANT_DIG);

    size_t nbaxis, nbpoints;
    std::string f1Header{}, f2Header{};
    std::ifstream file1(argv[1]);
    std::ifstream file2(argv[2]);
    double *m = readCSV(file1, f1Header, &nbaxis, &nbpoints);

    std::cerr << nbaxis << " " << nbpoints << std::endl;
    double *mean = mean_axises(m, nbaxis, nbpoints);
    std::cout << "Mean:" << std::endl;
    for (size_t i = 0; i < nbaxis; ++i)
    {
        std::cout << mean[i] << '\t';
    }
    std::cout << std::endl;
    free(mean);
    free(m);
    return EXIT_SUCCESS;
}