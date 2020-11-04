#include <float.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>

#include "libCSV/csv.hpp"
#include "libalg/print.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/CPUNumber.hpp"

int main(int argc, char *argv[])
{
    if (argc != 3)
        return EXIT_FAILURE;
    std::cout << std::setprecision(15); //DBL_MANT_DIG);

    size_t Pdim0, Pdim1;
    size_t Qdim0, Qdim1;
    std::string h{};
    double *array = readCSV(argv[1], h, Pdim0, Pdim1);
    double *Qarray = readCSV(argv[2], h, Qdim0, Qdim1);
    auto P = CPUMatrix{array, Pdim0, Pdim1};
    auto Q = CPUMatrix{Qarray, Qdim0, Qdim1};

    std::cerr << P << std::endl;
    std::cerr << P.mean(-1) << std::endl;
    std::cerr << P.mean(0) << std::endl;
    std::cerr << P.mean(1) << std::endl;

    std::cerr << P.sum(-1) << std::endl;
    std::cerr << P.sum(0) << std::endl;
    std::cerr << P.sum(1) << std::endl;
    return EXIT_SUCCESS;
}