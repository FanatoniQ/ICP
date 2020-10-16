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
#include "libalg/CPUMatrix.hpp"
#include "libalg/CPUNumber.hpp"

int main(int argc, char *argv[])
{
    if (argc != 3)
        return EXIT_FAILURE;
    std::cout << std::setprecision(15); //DBL_MANT_DIG);
    size_t Pdim0, Pdim1;
    std::string h{};
    double *array = readCSV(argv[1], h, Pdim0, Pdim1);

    print_matrix(std::cerr, array, Pdim1, Pdim0);
    auto P = CPUMatrix{array, Pdim0, Pdim1};
    std::cerr << P << std::endl;

    // Copy constructor is not allowed !!
    auto Pcopy = std::move(P); //CPUMat{P}; is not allowed
    std::cerr << "Transposing ..." << std::endl;
    auto P_T = Pcopy.transpose();
    std::cerr << "End Transposing ..." << std::endl;
    std::cerr << P_T << std::endl;

    auto meanP = Pcopy.mean(0);
    std::cerr << meanP << std::endl;
    std::cerr << Pcopy.mean(1) << std::endl;

    std::cerr << "*=5" << std::endl;
    std::cerr << meanP * CPUNumber(5) << std::endl;
    std::cerr << (Pcopy * CPUNumber(5)).mean(0) << std::endl;

    size_t Qdim0, Qdim1;
    double *Qarray = readCSV(argv[2], h, Qdim0, Qdim1);
    auto Q = CPUMatrix{Qarray, Qdim0, Qdim1};
    Q = Q.transpose(); // TODO: kinda bad... allocates even if not necessary (operator=)
    std::cerr << "Q transposed :" << std::endl;

    std::cerr << "P.dot(Q)" << std::endl;
    std::cerr << Pcopy.dot(Q) << std::endl;

    std::cerr << Pcopy << std::endl;
}