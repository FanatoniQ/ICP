#include <stdlib.h>

#include <iostream>
#include <fstream>

#include "libCSV/csv.hpp"
#include "error.hpp"
#include "libalg/alg.hpp"
#include "libalg/print.hpp"

int main(int argc, char *argv[])
{
    runtime_assert(argc == 2, "Usage: ./testlibalgcsvprint file1");
    size_t nbaxis, nbpoints;
    std::string f1Header{};
    float *m = readCSV(argv[1], f1Header, nbpoints, nbaxis);

    std::cerr << "nbaxis: " << nbaxis << " nbpoints: " << nbpoints << std::endl;
    print_matrix(std::cout, m, nbaxis, nbpoints);

    //runtime_assert(nbaxis > nbpoints, "fail checking");

    free(m);
    return 0;
}