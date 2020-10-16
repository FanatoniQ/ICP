#include <err.h> // errx

#include <float.h> // DBL_MANT_DIG: number of bits in mantisse
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <iomanip>

#include "libCSV/csv.hpp"
#include "libalg/alg.hpp"
#include "libalg/mean.hpp"
#include "cpu/icp.hpp"

int main(int argc, char *argv[])
{
    if (argc != 3)
        errx(1, "Usage: ./CPUICP file1 file2");
    std::cout << std::setprecision(15); //DBL_MANT_DIG);

    std::string f1Header{};
    size_t Qlines, Qcols, Plines, Pcols;
    //___readCSV(f, f1Header);
    double *P = readCSV(argv[1], f1Header, Plines, Pcols);
    double *Q = readCSV(argv[2], f1Header, Qlines, Qcols);
    /*
    if (!argv[1])
        return 1;
    Qlines = Qcols = Plines = Pcols = 2;
    double P[4] = {-2, -1, 5, 5};
    double Q[4] = {0, 1, 0, 1};
    */

    auto res = get_correspondence_indices(P, Q, Plines, Pcols, Qlines, Qcols);
    std::cout << std::get<0>(res.at(1)) << " and " << std::get<1>(res.at(1));
    //auto final = compute_cross_variance(P, Q, res, 2, 2, 2, 2, nullptr);
    //std::cout << std::get<0>(*final) << " and "<< std::get<1>(final.at(0));

    /*
    std::cerr << nblines << "x" << nbcols << " - " << f1Header << std::endl;
    for (size_t i = 0; i < nblines; ++i)
    {
        for (size_t j = 0; j < nbcols; ++j)
        {
            std::cerr << r[i * nbcols + j] << "\t";
        }
        std::cerr << std::endl;
    }
    */

    free(P);
    free(Q);

    /**
    size_t nbaxis, nbpoints;
    std::string f1Header{}, f2Header{};
    std::ifstream file1(argv[1]);
    std::ifstream file2(argv[2]);
    double *m = readCSV(file1, f1Header, &nbaxis, &nbpoints);

    std::cerr << "nbaxis: " << nbaxis << " nbpoints: " << nbpoints << std::endl;
    double *mean = mean_axises(m, nbaxis, nbpoints);
    std::cout << "Mean:" << std::endl;
    for (size_t i = 0; i < nbaxis; ++i)
    {
        std::cout << mean[i] << '\t';
    }
    std::cout << std::endl;

    std::cout << std::endl
              << "Centered:" << std::endl;
    element_wise_op(&m, m, mean, nbaxis, nbpoints, nbaxis, 1, subtract);
    for (size_t j = 0; j < nbpoints; ++j) // transposed printing, strided access
    {
        for (size_t i = 0; i < nbaxis; ++i)
        {
            std::cout << m[i * nbpoints + j] << '\t';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl
              << "Transposed:" << std::endl;
    double *m_T = transpose(m, nbaxis, nbpoints);
    for (size_t i = 0; i < nbpoints; ++i) // normal aligned access printing
    {
        for (size_t j = 0; j < nbaxis; ++j)
        {
            std::cout << m_T[i * nbaxis + j] << ',';
        }
        std::cout << std::endl;
    }
    std::cout << "(|[0,0,0] - P[:, 1]|_2)^2 = \t";
    double fakePoint = 0;
    double r = element_wise_reduce(&fakePoint, m_T + 3,
                                   1, 1, 1, nbaxis,
                                   squared_norm_2,
                                   add,
                                   add);
    std::cout << r << std::endl;

    std::cout << "P[:,1].dot(P[:,0].T) = ";
    double P1dotP0_T;
    double *P1dotP0_T_ptr = &P1dotP0_T; //(double *)calloc(1, sizeof(double));
    dot_product(&P1dotP0_T_ptr, m_T + nbaxis, m_T, 1, nbaxis, nbaxis, 1);
    std::cout << P1dotP0_T << std::endl;
    //free(P1dotP0_T);

    free(mean);
    free(m);
    free(m_T);
    **/
    return EXIT_SUCCESS;
}