#include "libalg/matrix.hpp"
#include <stdexcept>

Matrix::Matrix(): array(nullptr), dim0(0), dim1(0) {}

Matrix::Matrix(double* array, int dim0, int dim1) : array(array), dim0(dim0), dim1(dim1) {}

Matrix::Matrix(size_t dim0, size_t dim1, double init): dim0(dim0), dim1(dim1) {
    Matrix::array = (double*) malloc(dim0 * dim1 * sizeof(double));
    if (Matrix::array == nullptr)
        throw std::bad_alloc();
    for (unsigned i=0; i<dim0; ++i) {
        for (unsigned j=0; j<dim1; ++j) {
            Matrix::array[i * dim1 + j] = init;
        }
    }
}

std::vector<double> Matrix::mean_axis(size_t dim)
{
    size_t i, j;
    std::vector<double> r;
    if (dim == 0) {
        r.resize(dim1, 0);
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
        r.resize(dim0, 0);
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

Matrix::~Matrix() {
    free(array);
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
    Matrix result(dim0, dim1, 0.0);

    for (unsigned i=0; i<dim0; ++i) {
        for (unsigned j=0; j<dim1; ++j) {
            result(i,j) = (*this)(i,j) + rhs(i,j);
        }
    }
    return result;
}

Matrix &Matrix::operator+=(const Matrix &rhs) {

    for (unsigned i=0; i<rhs.getDim0(); ++i) {
        for (unsigned j=0; j<rhs.getDim1(); ++j) {
            (*this)(i,j) += rhs(i,j);
        }
    }
    return *this;
}

// FIXME be able to set with the operator
double &Matrix::operator()(const unsigned int &row, const unsigned int &col) {
    return array[row * dim0 + col];
}

const double &Matrix::operator()(const unsigned int &row, const unsigned int &col) const {
    return array[row * dim0 + col];
}



