#include <err.h>

#include <stdlib.h>

#include <iostream>
#include <fstream>

#include "libCSV/csv.hpp"
#include "libalg/alg.hpp"
#include "libalg/print.hpp"

int main(int argc, char *argv[])
{
    if (argc != 2)
        errx(1, "Usage: ./testlibalgcsvprint file1");
    size_t nbaxis, nbpoints;
    std::string f1Header{};
    double *m = readCSV(argv[1], f1Header, nbpoints, nbaxis);

    std::cerr << "nbaxis: " << nbaxis << " nbpoints: " << nbpoints << std::endl;
    print_matrix(std::cout, m, nbaxis, nbpoints);

    free(m);
    return 0;
}