#include "libgpualg/svd.cuh"
#include "error.cuh"
#include "error.hpp"
#include <cstdio>
#include <cstring>
#include <stdlib.h>
#include <tuple>
#include "libCSV/csv.hpp"
#include "libalg/CPUMatrix.hpp"

int main(int argc, char** argv) 
{
	runtime_assert(argc == 2, "Usage: ./testgpusum file1");

	size_t Pdim0, Pdim1;
	std::string h{};
	double* array = readCSV(argv[1], h, Pdim0, Pdim1);
	auto P = CPUMatrix{ array, Pdim0, Pdim1 };

	auto [U, S, V_T] = svd(array, Pdim0, Pdim1);

	std::cout << "Done GPU!";

	auto [V_T_cpu, S_cpu, U_cpu] = P.svd();

	std::cout << "U matrix CPU:" << U_cpu;
	std::cout << "S matrix CPU:" << S_cpu;
	std::cout << "VT matrix CPU:" << V_T_cpu;

	free(U);
	free(S);
	free(V_T);
}