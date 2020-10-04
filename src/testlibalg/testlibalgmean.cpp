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
    if (argc != 2)
        errx(1, "Usage: ./testlibalgmean file1");
    size_t nbaxis, nbpoints;
    std::string f1Header{};
    std::ifstream file1(argv[1]);
    double *m = readCSV(file1, f1Header, &nbaxis, &nbpoints);
    double *mean = mean_axises(m, nbaxis, nbpoints);

    std::cout << std::setprecision(15);
    std::cerr << "nbaxis: " << nbaxis << " nbpoints: " << nbpoints << std::endl;
    std::cerr << "Mean:" << std::endl;
    for (size_t i = 0; i < nbaxis; ++i)
    {
        std::cout << mean[i];
        if (i != nbaxis - 1)
            std::cout << ',';
    }
    std::cout << std::endl;

    free(mean);
    free(m);
    return 0;
}