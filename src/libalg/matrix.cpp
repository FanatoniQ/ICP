#include "libalg/matrix.hpp"

Matrix::Matrix(): array(nullptr), dim0(0), dim1(0) {}

Matrix::Matrix(double* array, int dim0, int dim1) : array(array), dim0(dim0), dim1(dim1) {}

std::vector<double> Matrix::mean_axis(size_t dim)
{
    size_t i, j;
    std::vector<double> r(5);

    for (i = 0; i < nb_axis; ++i)
    {
        for (j = 0; j < nb_points; ++j)
        {
            r[i] += m[i * nb_points + j];
        }
        r[i] /= nb_points;
    }
    return r;
}

bool Matrix::operator==(const Matrix &rhs) const {
    // TODO
    return array == rhs.array &&
           dim0 == rhs.dim0 &&
           dim1 == rhs.dim1;
}

bool Matrix::operator!=(const Matrix &rhs) const {
    // TODO
    return !(rhs == *this);
}

std::ostream &operator<<(std::ostream &os, const Matrix &matrix) {
    os << "array: " << matrix.array << " dim0: " << matrix.dim0 << " dim1: " << matrix.dim1;
    return os;
}

double *Matrix::getArray() const {
    return array;
}

void Matrix::setArray(double *array_) {
    Matrix::array = array_;
}

size_t Matrix::getDim0() const {
    return dim0;
}

void Matrix::setDim0(size_t dim0_) {
    Matrix::dim0 = dim0_;
}

size_t Matrix::getDim1() const {
    return dim1;
}

void Matrix::setDim1(size_t dim1_) {
    Matrix::dim1 = dim1_;
}

