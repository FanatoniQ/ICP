#pragma once

#include <ostream>
#include <vector>
#include <tuple>

class CPUView;

class CPUMatrix
{
protected:
    double *array;
    size_t dim0;
    size_t dim1;

public:
    // Constructors
    CPUMatrix();
    CPUMatrix(double *array, size_t dim0, size_t dim1);
    /**
     * Initialize matrix of dim0*dim1 with 0s
     * Throws std::bad_alloc on failed malloc
     * @param dim0
     * @param dim1
     */
    CPUMatrix(size_t dim0, size_t dim1);
    CPUMatrix(CPUMatrix const &mat) = delete; // avoid double free
    CPUMatrix(CPUMatrix &&mat) noexcept ;               // move constructor
    ~CPUMatrix();

    // Getters and setters
    double *getArray() const;

    /**
     * Replace the array pointer and returns the previous one
     * @param array_
     * @param dim0_
     * @param dim1_
     * @return CPUMatrix::array
    */
    double *setArray(double *array, size_t dim0_, size_t dim1_);
    size_t getDim0() const;
    size_t getDim1() const;

    // Output
    friend std::ostream &operator<<(std::ostream &os, const CPUMatrix &matrix);

    // Operations
    bool operator==(const CPUMatrix &rhs) const;
    bool operator!=(const CPUMatrix &rhs) const;
    CPUMatrix sum(int axis = 0);
    CPUMatrix mean(int axis = 0);
    double norm(const CPUMatrix &rhs);

    // Operator overloading, for "standard" mathematical matrix operations
    CPUMatrix &operator=(const CPUMatrix &rhs);

    // CPUMatrix mathematical operations
    CPUMatrix operator+(const CPUMatrix &rhs);
    CPUMatrix &operator+=(const CPUMatrix &rhs);
    CPUMatrix operator-(const CPUMatrix &rhs);
    CPUMatrix &operator-=(const CPUMatrix &rhs);
    CPUMatrix operator*(const CPUMatrix &rhs);
    CPUMatrix &operator*=(const CPUMatrix &rhs);
    CPUMatrix operator/(const CPUMatrix &rhs);
    CPUMatrix &operator/=(const CPUMatrix &rhs);

    CPUMatrix operator+(const double &rhs);
    CPUMatrix &operator+=(const double &rhs);
    CPUMatrix operator-(const double &rhs);
    CPUMatrix &operator-=(const double &rhs);
    CPUMatrix operator*(const double &rhs);
    CPUMatrix &operator*=(const double &rhs);
    CPUMatrix operator/(const double &rhs);
    CPUMatrix &operator/=(const double &rhs);

    CPUMatrix transpose();

    CPUMatrix dot(const CPUMatrix &rhs);

    std::tuple<CPUMatrix, CPUMatrix, CPUMatrix> svd();

    CPUView getLine(unsigned linenum);

    CPUMatrix copyLine(unsigned linenum);

    // Access the individual elements
    double &operator()(const unsigned &row, const unsigned &col);
    const double &operator()(const unsigned &row, const unsigned &col) const;
};
