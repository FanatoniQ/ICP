#include <stdlib.h>

#include <iostream>
#include <fstream>

#include "libCSV/csv.hpp"
#include "error.hpp"
#include "libalg/print.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/CPUNumber.hpp"
#include "libalg/CPUView.hpp"

int main(int argc, char *argv[])
{
    runtime_assert(argc == 3, "Usage: ./testview file1 file2");
    size_t nbaxis, nbpoints;
    std::string f1Header{};
    double *m = readCSV(argv[1], f1Header, nbpoints, nbaxis);
    size_t nbaxis1, nbpoints1;
    double *m1 = readCSV(argv[2], f1Header, nbpoints1, nbaxis1);
    auto P = CPUMatrix(m, nbpoints, nbaxis) * 2;
    //P = P + CPUMatrix(m1, nbpoints1, nbaxis1);
    P = P * 2;
    //std::cout << P * 2 << std::endl;
    P /= 4;
    //P *= CPUNumber(2);

    auto L = P.getLine(4); // L must be used before any reference changes to P
    //auto L = P.copyLine(4);
    P = CPUMatrix(m1, nbpoints1, nbaxis1); // this breaks L if called with another file since reference changes (we free the old array used by L)
    //auto L = P.copyLine(4);

    std::cout << P << std::endl;
    print_matrix(std::cout, P.getArray(), P.getDim1(), P.getDim0()); // watch out since P might change, do not use original pointer !

    std::cout << P.mean(0) << P.mean(1) << P.sum(0) / P.getDim0() << std::endl;
    std::cout << L << std::endl;

    std::cout << P.getLine(1) << std::endl;
    //m[0] *= 2.0;
    std::cout << L << std::endl;
    std::cerr << "P[:, 0].dot(P[:, 1].T)" << std::endl;
    std::cout << L.dot(P.getLine(1).transpose()) << std::endl;

    return 0;
}