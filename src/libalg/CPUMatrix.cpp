#include <stdexcept>

#include "libalg/alg.hpp"
#include "libalg/print.hpp"
#include "libalg/CPUMatrix.hpp"

// no need to free for user
CPUMatrix::CPUMatrix() : array(nullptr), dim0(0), dim1(0) {}

// no need to free for user
CPUMatrix::CPUMatrix(double *array, size_t dim0, size_t dim1) : array(array), dim0(dim0), dim1(dim1) {}

// no need to free for user
CPUMatrix::CPUMatrix(size_t dim0, size_t dim1)
{
    std::cerr << "Alloc !" << std::endl;
    this->array = (double *)calloc(dim0 * dim1, sizeof(double));
    if (this->array == nullptr)
        throw std::bad_alloc();
    this->dim0 = dim0;
    this->dim1 = dim1;
}

CPUMatrix::CPUMatrix(CPUMatrix &&mat) : array(mat.array), dim0(mat.dim0), dim1(mat.dim1)
{
    std::cerr << "Moved !" << std::endl;
    mat.array = nullptr;
}

/**
// Double free, we shall use a CPUMatrixViewClass instead or something
// all + - / * functions, basically functions returning a CPUMatrix by
// copy uses this... we should fix this by smart pointers or
// returning at the end the create matrix, meaning having to alloc
// the array... :'(
CPUMatrix::CPUMatrix(const CPUMatrix &mat)
{
    if (!mat.is_weak)
        std::runtime_error("Copy constructor can only be called with weak array !");
    this->array = mat.getArray();
    this->dim0 = mat.getDim0();
    this->dim1 = mat.getDim1();
    this->is_weak = false;
    // avoid double free (mat becomes a weak view of copy)
    //mat.array = nullptr; // TODO: issue
}**/

CPUMatrix::~CPUMatrix()
{
    if (array != nullptr)
        free(array);
}

CPUMatrix CPUMatrix::mean_axis(size_t dim)
{
    size_t i, j;
    size_t dimension = dim == 0 ? dim1 : dim0;
    CPUMatrix mat(1, dimension);

    // TODO: all this should be in low level API
    if (dim == 0)
    {
        for (i = 0; i < dim1; ++i)
        {
            for (j = 0; j < dim0; ++j)
            {
                mat.array[i] += array[j * dim1 + i];
            }
            mat.array[i] /= dim0;
        }
    }
    else
    {
        for (i = 0; i < dim0; ++i)
        {
            for (j = 0; j < dim1; ++j)
            {
                mat.array[i] += array[i * dim1 + j];
            }
            mat.array[i] /= dim1;
        }
    }
    return mat;
}

bool CPUMatrix::operator==(const CPUMatrix &rhs) const
{
    if (dim0 != rhs.dim0 || dim1 != rhs.dim1)
        return false;
    // TODO: memcmp ?
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
    //**
    for (unsigned i = 0; i < matrix.dim0; ++i)
    {
        for (unsigned j = 0; j < matrix.dim1; ++j)
        {
            os << matrix(i, j) << " ";
        }
        os << std::endl;
    }
    return os;
    //print_matrix(os, matrix.array, matrix.dim1, matrix.dim0);
    //return os;
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

// TODO: this is bad kinda, why not weak ptr copy and not freeing ?
CPUMatrix &CPUMatrix::operator=(const CPUMatrix &rhs)
{
    if (&rhs == this)
        return *this;

    unsigned new_rows = rhs.getDim0();
    unsigned new_cols = rhs.getDim1();

    // Allocate before freeing in case of error
    auto *new_array = (double *)malloc(new_rows * new_cols * sizeof(double));
    if (new_array == nullptr)
    {
        throw std::bad_alloc();
    }

    // Free old pointer
    free(array);
    array = new_array;
    dim0 = new_rows;
    dim1 = new_cols;

    for (unsigned i = 0; i < dim0; ++i)
    {
        for (unsigned j = 0; j < dim1; ++j)
        {
            (*this)(i, j) = rhs(i, j);
        }
    }
    return *this;
}

CPUMatrix CPUMatrix::operator+(const CPUMatrix &rhs)
{
    CPUMatrix result(dim0, dim1);
    // Add operation on both matrices
    element_wise_op(&result.array, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, add);
    return result;
}

CPUMatrix &CPUMatrix::operator+=(const CPUMatrix &rhs)
{
    // Add in-place operation
    element_wise_op(&this->array, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, add);
    return *this;
}

// FIXME be able to set with the operator / Maybe it's actually already the case
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
    CPUMatrix result(dim0, dim1);
    // Subtract operation on both matrices
    element_wise_op(&result.array, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, subtract);
    return result;
}

CPUMatrix &CPUMatrix::operator-=(const CPUMatrix &rhs)
{
    // Subtract in-place operation
    element_wise_op(&this->array, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, subtract);
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
    CPUMatrix result(dim0, dim1);
    // Multiply operation on both matrices
    element_wise_op(&result.array, this->array, rhs.array, this->dim0, this->dim1, rhs.dim0, rhs.dim1, mult);
    return result;
}

CPUMatrix &CPUMatrix::operator*=(const CPUMatrix &rhs)
{
    CPUMatrix result = (*this) * rhs;
    // TODO something cleaner?
    (*this) = result;
    return *this;
}

// TODO: FIX the low level transpose function
CPUMatrix CPUMatrix::transpose()
{
    double *r = ::transpose(array, dim0, dim1);
    /** Which constructor is called here ? **/
    CPUMatrix result;
    result.setArray(r, dim1, dim0);
    return result;
}
