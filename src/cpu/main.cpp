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
    //___readCSV(f, f1Header);
    double *Pt = readCSV(argv[1], f1Header, Plines, Pcols);
    CPUMatrix P = CPUMatrix(Pt, Plines, Pcols);

    double *Qt = readCSV(argv[2], f1Header, Qlines, Qcols);
    CPUMatrix Q = CPUMatrix(Qt, Qlines, Qcols);

    CPUMatrix refQ;
    refQ = Q;
    /*
    if (!argv[1])
        return 1;
    Qlines = Qcols = Plines = Pcols = 2;
    double P[4] = {-2, -1, 5, 5};
    double Q[4] = {0, 1, 0, 1};
    */

    //auto results = icp(P, Q, nbiters);
    //std::cout << "Found P: " << std::get<0>(results) << std::endl;
    //std::cout << "Ref Q: " << refQ << std::endl;
    //std::cout << "Squared mean diff: " << std::get<1>(results).back() << std::endl;
    //std::cout << "Squared actual mean diff: " << refQ.euclidianDistance(std::get<0>(results));
    //auto res = get_correspondence_indices(P, Q);
    std::tuple<size_t, int> *res2 = (std::tuple<size_t, int> *)calloc(Plines, sizeof(std::tuple<size_t, int>));
    
    get_correspondence_indices_array(res2, P.getArray(), Q.getArray(), Plines, Pcols, Qlines, Qcols);
    for (int i = 0; i < 30; i++)
        std::cout << std::get<0>(res2[i]) << "  " << std::get<1>(res2[i]) << std::endl;
        //std::cout << std::get<1>(res.at(i)) << "  " << std::get<1>(res2.at(i)) << std::endl;
    //free(res2);
    std::cout << "tout va bien" << std::endl;
    double *cov = (double *)calloc(9, sizeof(double));
    compute_cross_variance_array(cov, P.getArray(), Q.getArray(), res2, Plines, Pcols, Qlines, Qcols);
    free(res2);
    //std::cout << std::get<0>(final); //<< " and "<< std::get<1>(final).at(0);
    //auto finale = compute_cross_variance(P.getArray(), Q.getArray(), res2, P.getDim0(), P.getDim1(), Q.getDim0(), Q.getDim1(), nullptr);
    //auto arr = std::get<0>(finale);
    for (int i = 0; i < 9; i++)
        std::cout << *(cov+i) << std::endl;
    free(cov);
    //free(arr);
    //std::cout << "Errors:" << std::endl;
    //for (const auto &v : std::get<1>(results))
    //    std::cout << v << std::endl;
    //std::cout << std::get<0>(results) << std::endl;
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