#include "libgpualg/svd.cuh"
#include "error.cuh"
#include "error.hpp"
#include <cstdio>
#include <cstring>
#include <stdlib.h>
#include <tuple>
#include "libCSV/csv.hpp"
#include "libalg/CPUMatrix.hpp"

int checkMatrix(float *A, float *B, size_t dim0, size_t dim1)
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
	float* array = readCSV(argv[1], h, Pdim0, Pdim1);
	auto P = CPUMatrix{ array, Pdim0, Pdim1 };

	// m = row_A, n = col_A
	// (row * col)
	// U (m * m)
	// S (n)
	// VT (m * n)
	float *dA, 
		*dS, *dU, *dVt, 
		*S, *U, *VT;
	U = (float*)malloc(P.getDim1() * P.getDim1() * sizeof(float));
	S = (float*)malloc(P.getDim1() * P.getDim1() * sizeof(float));
	VT = (float*)malloc(P.getDim1() * P.getDim1() * sizeof(float));
	cudaMalloc(&dA, P.getDim0() * P.getDim1() * sizeof(float));
	cudaMalloc(&dU, P.getDim1() * P.getDim1() * sizeof(float));
	cudaMalloc(&dS, P.getDim1() * P.getDim1() * sizeof(float)); // FIXME is it rly the good shape
	cudaMalloc(&dVt, P.getDim1() * P.getDim1() * sizeof(float));

	cudaMemcpy(dA, array, P.getDim0() * P.getDim1() * sizeof(float), cudaMemcpyHostToDevice);

	svd_gpu(dA, Pdim0, Pdim1, dU, dS, dVt);

	std::cout << "Done GPU!";

	cudaMemcpy(U, dU, P.getDim1() * P.getDim1() * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(S, dS, P.getDim1() * P.getDim1() * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(VT, dVt, P.getDim1() * P.getDim1() * sizeof(float), cudaMemcpyDeviceToHost);

	auto [V_T_cpu, S_cpu, U_cpu] = P.svd();

	std::cout << "U matrix CPU:" << U_cpu;
	std::cout << "S matrix CPU:" << S_cpu;
	std::cout << "VT matrix CPU:" << V_T_cpu;

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