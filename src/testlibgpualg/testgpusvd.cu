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
			std::cerr << "Error at indice: " << i << "; Element " << A[i] << " is different from " << B[i] << std::endl;
			return 0;
		}
	}
	return 1;
}

int main(int argc, char** argv) 
{
	runtime_assert(argc == 2, "Usage: ./testgpusvd file1");

	size_t Pdim0, Pdim1;
	std::string h{};
	double* array = readCSV(argv[1], h, Pdim0, Pdim1);
	auto P = CPUMatrix{ array, Pdim0, Pdim1 };

	// m = row_A, n = col_A
	// (row * col)
	// U (m * m)
	// S (n)
	// VT (m * n)
	double *dA, 
		*dS, *dU, *dVt, 
		*S, *U, *VT;
	U = (double*)malloc(P.getDim1() * P.getDim1() * sizeof(double));
	S = (double*)malloc(P.getDim1() * P.getDim1() * sizeof(double));
	VT = (double*)malloc(P.getDim1() * P.getDim1() * sizeof(double));
	cudaMalloc(&dA, P.getDim0() * P.getDim1() * sizeof(double));
	cudaMalloc(&dU, P.getDim1() * P.getDim1() * sizeof(double));
	cudaMalloc(&dS, P.getDim1() * P.getDim1() * sizeof(double)); // FIXME is it rly the good shape
	cudaMalloc(&dVt, P.getDim1() * P.getDim1() * sizeof(double));

	cudaMemcpy(dA, array, P.getDim0() * P.getDim1() * sizeof(double), cudaMemcpyHostToDevice);

	svd_gpu(dA, Pdim0, Pdim1, dU, dS, dVt);

	std::cerr << "Done GPU!";

	cudaMemcpy(U, dU, P.getDim1() * P.getDim1() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(S, dS, P.getDim1() * P.getDim1() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(VT, dVt, P.getDim1() * P.getDim1() * sizeof(double), cudaMemcpyDeviceToHost);

	auto [V_T_cpu, S_cpu, U_cpu] = P.svd();

	std::cerr << "U matrix CPU:" << U_cpu;
	std::cerr << "S matrix CPU:" << S_cpu;
	std::cerr << "VT matrix CPU:" << V_T_cpu;

	runtime_assert(checkMatrix(U, U_cpu.getArray(), Pdim0, Pdim0), "Check matrix U failed \n");
	runtime_assert(checkMatrix(S, S_cpu.getArray(), Pdim1, 1), "Check matrix S failed \n");
	runtime_assert(checkMatrix(VT, V_T_cpu.getArray(), Pdim0, Pdim1), "Check matrix VT failed \n");

	cudaFree(dA);
	cudaFree(dS);
	cudaFree(dU);
	cudaFree(dVt);
	free(U);
	free(S);
	free(VT);
}