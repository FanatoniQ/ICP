#include "libalg/matrix.hpp"
#include <stdexcept>
#include "libalg/alg.hpp"

Matrix::Matrix(): array(nullptr), dim0(0), dim1(0) {}

Matrix::Matrix(double* array, size_t dim0, size_t dim1) : array(array), dim0(dim0), dim1(dim1) {}

Matrix::Matrix(size_t dim0, size_t dim1) {
    this->array = (double *)calloc(dim0 * dim1, sizeof(double));
    if (this->array == nullptr)
        throw std::bad_alloc();
    this->dim0 = dim0;
    this->dim1 = dim1;
}

double *Matrix::mean_axis(size_t dim)
{
    size_t i, j;
    double *r;
    if (dim == 0)
        r = (double *)calloc(dim1 , sizeof(double));
    else
        r = (double *)calloc(dim0 , sizeof(double));
    if (r == nullptr)
        throw std::bad_alloc();

    if (dim == 0) {
        for (i = 0; i < dim1; ++i)
        {
            for (j = 0; j < dim0; ++j)
            {
                r[i] += array[j * dim1 + i];
            }
            r[i] /= dim0;
        }
    }
    else {
        for (i = 0; i < dim0; ++i)
        {
            for (j = 0; j < dim1; ++j)
            {
                r[i] += array[i * dim1 + j];
            }
            r[i] /= dim1;
        }
    }
    return r;
}

bool Matrix::operator==(const Matrix &rhs) const {
    if (dim0 != rhs.dim0 || dim1 != rhs.dim1)
        return false;
    for (unsigned i=0; i<dim0; ++i) {
        for (unsigned j=0; j<dim1; ++j) {
            if ((*this)(i,j) != rhs(i, j))
                return false;
        }
    }
    return true;
}

bool Matrix::operator!=(const Matrix &rhs) const {
    return !(rhs == *this);
}

std::ostream &operator<<(std::ostream &os, const Matrix &matrix) {
    os << " dim0: " << matrix.dim0 << " dim1: " << matrix.dim1;
    os << "array: ";
    for (unsigned i=0; i<matrix.dim0; ++i) {
        for (unsigned j=0; j<matrix.dim1; ++j) {
            os << matrix(i, j) << " ";
        }
    }
    return os;
}

double *Matrix::getArray() const {
    return array;
}

double *Matrix::setArray(double *array_, size_t dim0_, size_t dim1_) {
    double *tmp = Matrix::array;
    Matrix::array = array_;
    Matrix::dim0 = dim0_;
    Matrix::dim1 = dim1_;
    return tmp;
}

size_t Matrix::getDim0() const {
    return dim0;
}

size_t Matrix::getDim1() const {
    return dim1;
}

Matrix &Matrix::operator=(const Matrix &rhs) {
    if (&rhs == this)
        return *this;

    unsigned new_rows = rhs.getDim0();
    unsigned new_cols = rhs.getDim1();

    // Allocate before freeing in case of error
    auto *new_array = (double*)malloc(new_rows * new_cols * sizeof(double));
    if (new_array == nullptr) {
        throw;
    }

    // Free old pointer
    free(array);
    array = new_array;
    dim0 = new_rows;
    dim1 = new_cols;

    for (unsigned i=0; i<dim0; ++i) {
        for (unsigned j=0; j<dim1; ++j) {
            (*this)(i,j) = rhs(i, j);
        }
    }
    return *this;
}

Matrix Matrix::operator+(const Matrix &rhs) {
    Matrix result(dim0, dim1);
    // Add operation on both matrices
    element_wise_op(&result.array, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, add);
    return result;
}

Matrix &Matrix::operator+=(const Matrix &rhs) {
    // Add in-place operation
    element_wise_op(&this->array, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, add);
    return *this;
}

// FIXME be able to set with the operator / Maybe it's actually already the case
double &Matrix::operator()(const unsigned int &row, const unsigned int &col) {
    return array[row * dim0 + col];
}

const double &Matrix::operator()(const unsigned int &row, const unsigned int &col) const {
    return array[row * dim0 + col];
}

Matrix Matrix::operator-(const Matrix &rhs) {
    Matrix result(dim0, dim1);
    // Subtract operation on both matrices
    element_wise_op(&result.array, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, subtract);
    return result;
}

Matrix &Matrix::operator-=(const Matrix &rhs) {
    // Subtract in-place operation
    element_wise_op(&this->array, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, subtract);
    return *this;
}

Matrix Matrix::operator*(const Matrix &rhs) {
    Matrix result(this->dim0, rhs.dim1);
    dot_product(&result.array, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1);
    return result;
}

Matrix &Matrix::operator*=(const Matrix &rhs) {
    Matrix result = (*this) * rhs;
    // TODO something cleaner?
    (*this) = result;
    return *this;
}

Matrix Matrix::transpose() {
    double *r = ::transpose(array, dim0, dim1);
    double *tmp = this->setArray(r, dim1, dim0);
    free(tmp);
    return *this;
}




