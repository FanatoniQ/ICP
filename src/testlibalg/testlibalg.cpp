#include <float.h>
#include <stdlib.h>

#include <math.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>

#include "libCSV/csv.hpp"
#include "error.hpp"
#include "libalg/print.hpp"
#include "libalg/svd.hpp"
#include "libalg/alg.hpp"
#include "libalg/mean.hpp"
#include "libalg/CPUMatrix.hpp"

#define UNUSED(x) (void)x

int test_norm(char *file1, char *file2)
{
    size_t Pdim0, Pdim1;
    size_t Qdim0, Qdim1;
    std::string h{};
    float *Parray = readCSV(file1, h, Pdim0, Pdim1);
    float *Qarray = readCSV(file2, h, Qdim0, Qdim1);
    float dist = element_wise_reduce(Parray, Qarray, Pdim0, Pdim1, Qdim0, Qdim1, squared_norm_2, add, add);
    std::cout << sqrt(dist) << std::endl;
    free(Parray);
    free(Qarray);
    return EXIT_SUCCESS;
}

int test_svd(char *file)
{
    size_t nbaxis, nbpoints;
    std::string h{};
    float *a = readCSV(file, h, nbpoints, nbaxis);
    /** This needs a fix !
    auto R = CPUMatrix(a, nbpoints, nbaxis).svd();
    std::cerr << std::get<0>(R) << std::endl;
    std::cerr << std::get<1>(R) << std::endl;
    std::cerr << std::get<2>(R) << std::endl;
    std::cerr << (std::get<0>(R) * std::get<1>(R)).dot(std::get<2>(R)) << std::endl;
    **/
    int n = nbpoints, m = nbaxis;
    float *u = NULL, *sigma = NULL, *vt = NULL;
    int sizes;
    svd(a, &u, &sigma, &vt, m, n, &sizes);

    // Full matrices exemple:
    //print_matrix(std::cout, vt, n, n, n);        // n,n full matrices
    //print_matrix(std::cout, sigma, sizes, 1, 1); // 1,min(n,m)
    //print_matrix(std::cout, u, m, m, m);         // m,m full matrices

    // Not Full Matrices
    print_matrix(std::cout, vt, sizes, n, n);    // n,sizes not full matrices
    print_matrix(std::cout, sigma, sizes, 1, 1); // 1,sizes
    print_matrix(std::cout, u, m, sizes, m);     // m,m full matrices

    free(u);
    free(sigma);
    free(vt);
    free(a);
    return EXIT_SUCCESS;
}

int test_sum(char *file, int axis)
{
    size_t dim0, dim1;
    std::string f1Header{};

    float *m = readCSV(file, f1Header, dim0, dim1);
    float *mean = nullptr;
    size_t dimr;
    sum_axises(&mean, m, dim0, dim1, dimr, axis);
    std::cerr << "nbaxis: " << dim1 << " nbpoints: " << dim0 << std::endl;
    std::cerr << "Mean:" << std::endl;
    print_matrix(std::cout, mean, dimr, 1);
    free(m);
    free(mean);
    return EXIT_SUCCESS;
}

int test_mean(char *file, int axis)
{
    size_t dim0, dim1;
    std::string f1Header{};

    float *m = readCSV(file, f1Header, dim0, dim1);
    float *mean = nullptr;
    size_t dimr;
    mean_axises(&mean, m, dim0, dim1, dimr, axis);
    std::cerr << "nbaxis: " << dim1 << " nbpoints: " << dim0 << std::endl;
    std::cerr << "Mean:" << std::endl;
    print_matrix(std::cout, mean, dimr, 1);
    free(m);
    free(mean);
    return EXIT_SUCCESS;
}

int test_op(char *file1, char *file2, float (*op)(float a, float b))
{
    size_t Pdim0, Pdim1;
    size_t Qdim0, Qdim1;
    size_t Rdim0, Rdim1;
    std::string h{};
    float *Parray = readCSV(file1, h, Pdim0, Pdim1);
    float *Qarray = readCSV(file2, h, Qdim0, Qdim1);

    float *r = nullptr;
    element_wise_op(&r, Parray, Qarray, Pdim0, Pdim1, Qdim0, Qdim1, Rdim0, Rdim1, op);
    print_matrix(std::cout, r, Rdim1, Rdim0);

    free(Parray);
    free(Qarray);
    free(r);
    return EXIT_SUCCESS;
}

int test_transpose(char *file1)
{
    size_t Pdim0, Pdim1;
    std::string h{};
    float *Parray = readCSV(file1, h, Pdim0, Pdim1);
    float *P_Tarray = transpose(Parray, Pdim0, Pdim1);
    print_matrix(std::cout, P_Tarray, Pdim0, Pdim1);
    free(Parray);
    free(P_Tarray);
    return EXIT_SUCCESS;
}

int test_dotproduct(char *file1, char *file2)
{
    size_t Pdim0, Pdim1;
    size_t Qdim0, Qdim1;
    std::string h{};
    float *Parray = readCSV(file1, h, Pdim0, Pdim1);
    float *Qarray = readCSV(file2, h, Qdim0, Qdim1);

    auto P = CPUMatrix(Parray, Pdim0, Pdim1);
    auto Q = CPUMatrix(Qarray, Qdim0, Qdim1);
    Q = Q.transpose();

    auto R = P.dot(Q);
    print_matrix(std::cout, R.getArray(), R.getDim1(), R.getDim0());

    return EXIT_SUCCESS;

    /**
    size_t nbaxis, nbpoints;
    std::string f1Header{}, f2Header{};
    std::ifstream file1(argv[1]);
    std::ifstream file2(argv[2]);
    float *m = readCSV(file1, f1Header, &nbaxis, &nbpoints);
    float *m_T = transpose(m, nbaxis, nbpoints);
    float *n = readCSV(file2, f2Header, &nbaxis, &nbpoints);
    float *r = NULL;
    dot_product(&r, m_T, n, nbpoints, nbaxis, nbaxis, nbpoints);
    print_matrix(std::cout, r, nbpoints, nbpoints);
    free(m);
    free(r);
    free(n);
    free(m_T);
    return EXIT_SUCCESS; **/
}

void usage(void)
{
    std::cerr << "Usage:" << std::endl
              << std::endl;
    std::cerr << "./testlibalg add file1 file2" << std::endl;
    std::cerr << "./testlibalg subtract file1 file2" << std::endl;
    std::cerr << "./testlibalg mult file1 file2" << std::endl;
    std::cerr << "./testlibalg sum file1 axis=[-1,0,1]" << std::endl;
    std::cerr << "./testlibalg mean file1 axis=[-1,0,1]" << std::endl;
    std::cerr << "./testlibalg svd file1" << std::endl;
    std::cerr << "./testlibalg transpose file1" << std::endl;
    std::cerr << "./testlibalg dotproduct file1 file2" << std::endl
              << std::endl;
    exit(1);
}

int main(int argc, char *argv[])
{
    std::cout << std::setprecision(15); //DBL_MANT_DIG);
    if (argc == 3)
    {
        if (strcmp(argv[1], "svd") == 0)
            return test_svd(argv[2]);
        else if (strcmp(argv[1], "transpose") == 0)
            return test_transpose(argv[2]);
    }
    else if (argc == 4)
    {
        if (strcmp(argv[1], "add") == 0)
            return test_op(argv[2], argv[3], add);
        else if (strcmp(argv[1], "subtract") == 0)
            return test_op(argv[2], argv[3], subtract);
        else if (strcmp(argv[1], "mult") == 0)
            return test_op(argv[2], argv[3], mult);
        else if (strcmp(argv[1], "dotproduct") == 0)
            return test_dotproduct(argv[2], argv[3]);
        else if (strcmp(argv[1], "mean") == 0)
            return test_mean(argv[2], std::stoi(argv[3]));
        else if (strcmp(argv[1], "sum") == 0)
            return test_sum(argv[2], std::stoi(argv[3]));
        else if (strcmp(argv[1], "norm") == 0)
            return test_norm(argv[2], argv[3]);
    }
    usage();
}