#include "libgpualg/svd.cuh"
#include "error.cuh"
#include "error.hpp"
#include <cstdio>
#include <cstring>
#include <stdlib.h>
#include <tuple>
#include "libCSV/csv.hpp"
#include "libalg/CPUMatrix.hpp"

int checkMatrix(double *A, double *B, size_t dim0, size_t dim1)
{
	for (int i = 0; i < dim0 * dim1; ++i)
	{
		if (std::abs(A[i] - B[i]) > 0.01)
		{
			std::cout << "Error at indice: " << i << "; Element " << A[i] << " is different from " << B[i] << std::endl;
			return 0;
		}
	}
	return 1;
}

int main(int argc, char** argv) 
{
	runtime_assert(argc == 2, "Usage: ./testgpusum file1");

	size_t Pdim0, Pdim1;
	std::string h{};
	double* array = readCSV(argv[1], h, Pdim0, Pdim1);
	auto P = CPUMatrix{ array, Pdim0, Pdim1 };

	// m = row_A, n = col_A
	// (row * col)
	// U (m * m)
	// S (n)
	// VT (m * n)
	auto [U, S, V_T] = svd(array, Pdim0, Pdim1);

	std::cout << "Done GPU!";

	auto [V_T_cpu, S_cpu, U_cpu] = P.svd();

	std::cout << "U matrix CPU:" << U_cpu;
	std::cout << "S matrix CPU:" << S_cpu;
	std::cout << "VT matrix CPU:" << V_T_cpu;

	runtime_assert(checkMatrix(U, U_cpu.getArray(), Pdim0, Pdim0), "Check matrix U failed \n");
	runtime_assert(checkMatrix(S, S_cpu.getArray(), Pdim1, 1), "Check matrix S failed \n");
	runtime_assert(checkMatrix(V_T, V_T_cpu.getArray(), Pdim0, Pdim1), "Check matrix VT failed \n");

	free(U);
	free(S);
	free(V_T);
}