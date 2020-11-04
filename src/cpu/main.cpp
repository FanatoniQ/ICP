#include <float.h> // DBL_MANT_DIG: number of bits in mantisse
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <libalg/print.hpp>

#include "libCSV/csv.hpp"
#include "error.hpp"
#include "libalg/alg.hpp"
#include "libalg/mean.hpp"
#include "cpu/icp.hpp"

int main(int argc, char *argv[])
{
    runtime_assert(argc == 4, "Usage: ./CPUICP file1 file2 nbiters");
    //std::cout << std::setprecision(15); //DBL_MANT_DIG);
    int nbiters = std::stoi(argv[3]);
    runtime_assert(nbiters > 0, "nbiter <= 0");

    std::string f1Header{};
    size_t Qlines, Qcols, Plines, Pcols;

    double *Pt = readCSV(argv[1], f1Header, Plines, Pcols);
    CPUMatrix P = CPUMatrix(Pt, Plines, Pcols);

    double *Qt = readCSV(argv[2], f1Header, Qlines, Qcols);
    CPUMatrix Q = CPUMatrix(Qt, Qlines, Qcols);

    CPUMatrix refQ;
    refQ = Q;

    auto results = icp(P, Q, nbiters);
    CPUMatrix Pres = std::move(std::get<0>(results));
    std::cout << "Squared actual mean diff: " << refQ.euclidianDistance(Pres) << std::endl;
    Pres.write_in_file("point_cloud_result");

    return EXIT_SUCCESS;
}