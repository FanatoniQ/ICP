#include <cstring>

#include <stdexcept>
#include <cmath>

#include "error.hpp"
#include "libalg/alg.hpp"
#include "libalg/mean.hpp"
#include "libalg/print.hpp"
#include "libalg/basic_operations.hpp"
#include "libalg/svd.hpp"
#include "libalg/CPUMatrix.hpp"
#include "libalg/CPUView.hpp"
#include "libalg/CPUNumber.hpp"

// no need to free for user
CPUMatrix::CPUMatrix() : array(nullptr), dim0(0), dim1(0) {}

// no need to free for user
CPUMatrix::CPUMatrix(double *array, size_t dim0, size_t dim1) : array(array), dim0(dim0), dim1(dim1) {}

// no need to free for user
CPUMatrix::CPUMatrix(size_t dim0, size_t dim1)
{
    this->array = (double *)calloc(dim0 * dim1, sizeof(double));
    if (this->array == nullptr)
        throw std::bad_alloc();
    this->dim0 = dim0;
    this->dim1 = dim1;
}

// move constructor
CPUMatrix::CPUMatrix(CPUMatrix &&mat) noexcept : array(mat.array), dim0(mat.dim0), dim1(mat.dim1)
{
    mat.array = nullptr;
}

CPUMatrix::~CPUMatrix()
{
    if (array != nullptr)
        free(array);
}

CPUMatrix CPUMatrix::sum(int axis)
{
    size_t dimr;
    double *r = NULL;
    ::sum_axises(&r, array, dim0, dim1, dimr, axis);
    return CPUMatrix(r, 1, dimr);
}

CPUMatrix CPUMatrix::mean(int axis)
{
    size_t dimr;
    double *r = NULL;
    ::mean_axises(&r, array, dim0, dim1, dimr, axis);
    return CPUMatrix(r, 1, dimr);
}

bool CPUMatrix::operator==(const CPUMatrix &rhs) const
{
    if (dim0 != rhs.dim0 || dim1 != rhs.dim1)
        return false;
    for (unsigned i = 0; i < dim0; ++i)
    {
        for (unsigned j = 0; j < dim1; ++j)
        {
            if ((*this)(i, j) != rhs(i, j))
                return false;
        }
    }
    return true;
}

bool CPUMatrix::operator!=(const CPUMatrix &rhs) const
{
    return !(rhs == *this);
}

std::ostream &operator<<(std::ostream &os, const CPUMatrix &matrix)
{
    os << " dim0: " << matrix.dim0 << " dim1: " << matrix.dim1;
    os << " array: " << std::endl;
    for (unsigned i = 0; i < matrix.dim0; ++i)
    {
        for (unsigned j = 0; j < matrix.dim1; ++j)
        {
            os << matrix(i, j) << " ";
        }
        os << std::endl;
    }
    return os;
}

double *CPUMatrix::getArray() const
{
    return array;
}

double *CPUMatrix::setArray(double *array_, size_t dim0_, size_t dim1_)
{
    double *tmp = CPUMatrix::array;
    CPUMatrix::array = array_;
    CPUMatrix::dim0 = dim0_;
    CPUMatrix::dim1 = dim1_;
    return tmp;
}

size_t CPUMatrix::getDim0() const
{
    return dim0;
}

size_t CPUMatrix::getDim1() const
{
    return dim1;
}

CPUMatrix &CPUMatrix::operator=(const CPUMatrix &rhs)
{
    if (&rhs == this)
        return *this;
    bool noalloc = ((dim0 * dim1) == (rhs.getDim0() * rhs.getDim1()));
    dim0 = rhs.getDim0();
    dim1 = rhs.getDim1();

    if (!noalloc)
    {
        if (array != nullptr)
            free(array);
        this->array = (double *)malloc(dim0 * dim1 * sizeof(double));
        if (array == nullptr)
            throw std::bad_alloc();
    }
    memcpy(array, rhs.array, dim0 * dim1 * sizeof(double));
    return *this;
}

CPUMatrix CPUMatrix::operator+(const CPUMatrix &rhs)
{
    // Add operation on both matrices
    size_t Rdim0, Rdim1;
    double *r = nullptr;
    element_wise_op(&r, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, Rdim0, Rdim1, add);
    return CPUMatrix(r, Rdim0, Rdim1);
}

CPUMatrix &CPUMatrix::operator+=(const CPUMatrix &rhs)
{
    // Add in-place operation
    element_wise_op(&this->array, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, this->dim0, this->dim1, add);
    return *this;
}

double &CPUMatrix::operator()(const unsigned int &row, const unsigned int &col)
{
    return array[row * dim1 + col];
}

const double &CPUMatrix::operator()(const unsigned int &row, const unsigned int &col) const
{
    return array[row * dim1 + col];
}

CPUMatrix CPUMatrix::operator-(const CPUMatrix &rhs)
{
    // Subtract operation on both matrices
    size_t Rdim0, Rdim1;
    double *r = nullptr;
    element_wise_op(&r, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, Rdim0, Rdim1, subtract);
    return CPUMatrix(r, Rdim0, Rdim1);
}

CPUMatrix &CPUMatrix::operator-=(const CPUMatrix &rhs)
{
    // Subtract in-place operation
    element_wise_op(&this->array, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, this->dim0, this->dim1, subtract);
    return *this;
}

CPUMatrix CPUMatrix::dot(const CPUMatrix &rhs)
{
    CPUMatrix result(this->dim0, rhs.dim1);
    dot_product(&result.array, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1);
    return result;
}

CPUMatrix CPUMatrix::operator*(const CPUMatrix &rhs)
{
    // Multiply operation on both matrices
    size_t Rdim0, Rdim1;
    double *r = nullptr;
    element_wise_op(&r, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, Rdim0, Rdim1, mult);
    return CPUMatrix(r, Rdim0, Rdim1);
}

CPUMatrix &CPUMatrix::operator*=(const CPUMatrix &rhs)
{
    element_wise_op(&this->array, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, this->dim0, this->dim1, mult);
    return *this;
}

CPUMatrix CPUMatrix::operator/(const CPUMatrix &rhs)
{
    size_t Rdim0, Rdim1;
    double *r = nullptr;
    element_wise_op(&r, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, Rdim0, Rdim1, divide);
    return CPUMatrix(r, Rdim0, Rdim1);
}

CPUMatrix &CPUMatrix::operator/=(const CPUMatrix &rhs)
{
    element_wise_op(&this->array, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, this->dim0, this->dim1, divide);
    return *this;
}

CPUMatrix CPUMatrix::operator+(const double &rhs)
{
    return *this + CPUNumber(rhs);
}

CPUMatrix CPUMatrix::operator-(const double &rhs)
{
    return *this - CPUNumber(rhs);
}

CPUMatrix CPUMatrix::operator*(const double &rhs)
{
    return *this * CPUNumber(rhs);
}

CPUMatrix CPUMatrix::operator/(const double &rhs)
{
    return *this / CPUNumber(rhs);
}

CPUMatrix &CPUMatrix::operator+=(const double &rhs)
{
    *this += CPUNumber(rhs);
    return *this;
}

CPUMatrix &CPUMatrix::operator-=(const double &rhs)
{
    *this -= CPUNumber(rhs);
    return *this;
}

CPUMatrix &CPUMatrix::operator*=(const double &rhs)
{
    *this *= CPUNumber(rhs);
    return *this;
}

CPUMatrix &CPUMatrix::operator/=(const double &rhs)
{
    *this /= CPUNumber(rhs);
    return *this;
}

CPUMatrix CPUMatrix::transpose()
{
    double *r = ::transpose(array, dim0, dim1);
    CPUMatrix result;
    result.setArray(r, dim1, dim0);
    return result;
}

CPUView CPUMatrix::getLine(unsigned linenum)
{
    runtime_assert(linenum < dim0, "Invalid line number !");
    return CPUView(array + dim1 * linenum, dim1);
}

CPUMatrix CPUMatrix::copyLine(unsigned linenum)
{
    auto r = CPUMatrix(1, dim1);
    memcpy(r.array, array + dim1 * linenum, dim1 * sizeof(double));
    return r;
}

CPUMatrix CPUMatrix::squared_norm(int axis)
{
    double *r = nullptr;
    size_t dimr;
    reduce_axises(&r, array, dim0, dim1, dimr, axis, pow2, add);
    return CPUMatrix(r, 1, dimr);
}

double CPUMatrix::euclidianDistance(const CPUMatrix &rhs)
{
    double norm = element_wise_reduce(array, rhs.array, dim0, dim1, rhs.dim0, rhs.dim1, squared_norm_2, add, add);
    return sqrt(norm);
}

std::tuple<CPUMatrix, CPUMatrix, CPUMatrix> CPUMatrix::svd()
{
    size_t nbpoints = dim0, nbaxis = dim1;
    int n = nbpoints, m = nbaxis;
    double *u = nullptr, *sigma = nullptr, *vt = nullptr;
    int sizes;
    ::svd(array, &u, &sigma, &vt, m, n, &sizes);

    if (sizes == m)
    {
        // Linearize vt
        // vt nbols=sizes, nblines=n, ld=n
        double *VT = ::linearize(vt, nbpoints, sizes, nbpoints); // vt = shape(n,n)
        free(vt);
        vt = VT;
    }
    else if (sizes != n)
        runtime_failure("invalid sizes");
    std::cerr << "shapes: " << n << "," << n << "  " << n << ",  " << n << "," << m << std::endl;
    // VT : n,sizes not full matrices
    // Sigma : 1,sizes
    // U : m,m full matrices

    return std::make_tuple(CPUMatrix(vt, nbpoints, sizes), CPUMatrix(sigma, 1, sizes), CPUMatrix(u, sizes, nbaxis));
}