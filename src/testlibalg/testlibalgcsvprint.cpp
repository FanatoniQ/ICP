#include <err.h>

#include <stdlib.h>

#include <iostream>
#include <fstream>

#include "libCSV/csv.hpp"
#include "libalg/alg.hpp"

int main(int argc, char *argv[])
{
    if (argc != 2)
        errx(1, "Usage: ./testlibalgcsvprint file1");
    size_t nbaxis, nbpoints;
    std::string f1Header{};
    std::ifstream file1(argv[1]);
    double *m = readCSV(file1, f1Header, &nbaxis, &nbpoints);

    std::cerr << "nbaxis: " << nbaxis << " nbpoints: " << nbpoints << std::endl;
    for (size_t i = 0; i < nbpoints / nbaxis; ++i)
    {
        for (size_t j = 0; j < nbaxis; ++j)
        {
            std::cout << m[i*nbaxis + j];
            if (j != nbaxis - 1)
                std::cout << ',';
            else
                std::cout << std::endl;
            
        }
    }

    free(m);
    return 0;
}