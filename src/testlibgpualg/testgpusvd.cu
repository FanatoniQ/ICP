#include "libgpualg/svd.cuh"
#include "error.cuh"
#include "error.hpp"
#include <cstdio>
#include <cstring>
#include <stdlib.h>
#include <tuple>
#include "libCSV/csv.hpp"

int main(int argc, char** argv) 
{
	runtime_assert(argc == 2, "Usage: ./testgpusum file1");

	size_t Pdim0, Pdim1;
	std::string h{};
	double* P = readCSV(argv[1], h, Pdim0, Pdim1);

	auto [U, S, V_T] = svd(P, Pdim0, Pdim1);

	std::cout << "Done!";
	free(U);
	free(S);
	free(V_T);
}