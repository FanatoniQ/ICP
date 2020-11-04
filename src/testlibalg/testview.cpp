#include <stdlib.h>

#include <iostream>
#include <iomanip>
#include <fstream>

#include "libCSV/csv.hpp"
#include "error.hpp"
#include "libalg/print.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/CPUNumber.hpp"
#include "libalg/CPUView.hpp"

int main(int argc, char *argv[])
{
    runtime_assert(argc == 2, "Usage: ./testview file1");
    std::cerr << std::setprecision(15);
    size_t nbaxis, nbpoints;
    std::string f1Header{};
    double *m = readCSV(argv[1], f1Header, nbpoints, nbaxis);

    auto P = CPUMatrix(m, nbpoints, nbaxis); // * 2;

    auto R = P.svd(); // FIXME: we do not handle lda for now...
    //print_matrix(std::cout, vt, nbpoints, nbpoints, nbpoints); // not full matrices
    //print_matrix(std::cout, sigma, nbpoints, 1, 1);            // shape is: n,
    //print_matrix(std::cout, U, nbaxis, nbpoints);
    std::cerr << std::get<0>(R) << std::endl; // this should be linearized
    std::cerr << std::get<1>(R) << std::endl; // this should be linearized
    std::cerr << std::get<2>(R) << std::endl; // this should be linearized

    auto recovered = (std::get<0>(R) * std::get<1>(R)).dot(std::get<2>(R));

    std::cerr << recovered << std::endl;

    return 0;
}