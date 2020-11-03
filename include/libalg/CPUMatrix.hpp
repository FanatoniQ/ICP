#pragma once

#include <ostream>
#include <vector>
#include <tuple>

class CPUView;

class CPUMatrix
{
protected:
    float *array;
    size_t dim0;
    size_t dim1;

public:
    // Constructors
    CPUMatrix();
    CPUMatrix(float *array, size_t dim0, size_t dim1);
    /**
     * Initialize matrix of dim0*dim1 with 0s
     * Throws std::bad_alloc on failed malloc
     * @param dim0
     * @param dim1
     */
    CPUMatrix(size_t dim0, size_t dim1);
    CPUMatrix(CPUMatrix const &mat) = delete; // avoid float free
    CPUMatrix(CPUMatrix &&mat) noexcept;      // move constructor
    ~CPUMatrix();

    // Getters and setters
    float *getArray() const;

    /**
     * Replace the array pointer and returns the previous one
     * @param array_
     * @param dim0_
     * @param dim1_
     * @return CPUMatrix::array
    */
    float *setArray(float *array, size_t dim0_, size_t dim1_);
    size_t getDim0() const;
    size_t getDim1() const;

    // Output
    friend std::ostream &operator<<(std::ostream &os, const CPUMatrix &matrix);

    // Operations
    bool operator==(const CPUMatrix &rhs) const;
    bool operator!=(const CPUMatrix &rhs) const;
    CPUMatrix sum(int axis = 0);
    CPUMatrix mean(int axis = 0);
    float euclidianDistance(const CPUMatrix &rhs);

    // prefer using euclidianDistance when possible
    CPUMatrix squared_norm(int axis = -1);

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

    CPUMatrix operator+(const float &rhs);
    CPUMatrix &operator+=(const float &rhs);
    CPUMatrix operator-(const float &rhs);
    CPUMatrix &operator-=(const float &rhs);
    CPUMatrix operator*(const float &rhs);
    CPUMatrix &operator*=(const float &rhs);
    CPUMatrix operator/(const float &rhs);
    CPUMatrix &operator/=(const float &rhs);

    CPUMatrix transpose();

    CPUMatrix dot(const CPUMatrix &rhs);

    std::tuple<CPUMatrix, CPUMatrix, CPUMatrix> svd();

    CPUView getLine(unsigned linenum);
    CPUMatrix copyLine(unsigned linenum);

    // Access the individual elements
    float &operator()(const unsigned &row, const unsigned &col);
    const float &operator()(const unsigned &row, const unsigned &col) const;
};
