#include "libalg/CPUMat.hpp"

CPUMat::~CPUMat() {
    free(this->array);
}

CPUMat::CPUMat(size_t dim0, size_t dim1) : Matrix(dim0, dim1) {

}

CPUMat::CPUMat(double *array, size_t dim0, size_t dim1) : Matrix(array, dim0, dim1) {

}
