#pragma once

#include <ostream>
#include <vector>

class Matrix {
protected:
    double* array;
    size_t dim0;
    size_t dim1;
    // Constructors
    Matrix();
    Matrix(double *array, size_t dim0, size_t dim1);
    /**
     * Initialize matrix of dim0*dim1 with 0s
     * Throws std::bad_alloc on failed malloc
     * @param dim0
     * @param dim1
     */
    Matrix(size_t dim0, size_t dim1);
    Matrix(Matrix const &mat);

public:
    /**
     * Free the ressources
     */
    virtual ~Matrix();

    // Getters and setters
    double *getArray() const;

    /**
     * Replace the array pointer and returns the previous one
     * @param array_
     * @param dim0_
     * @param dim1_
     * @return Matrix::array
    */
    double *setArray(double *array, size_t dim0_, size_t dim1_);
    size_t getDim0() const;
    size_t getDim1() const;

    // Output
    friend std::ostream &operator<<(std::ostream &os, const Matrix &matrix);

    // Operations
    bool operator==(const Matrix &rhs) const;
    bool operator!=(const Matrix &rhs) const;
    Matrix mean_axis(size_t dim);

    // Operator overloading, for "standard" mathematical matrix operations
    Matrix& operator=(const Matrix& rhs);

    // Matrix mathematical operations
    Matrix operator+(const Matrix& rhs);
    Matrix& operator+=(const Matrix& rhs);
    Matrix operator-(const Matrix& rhs);
    Matrix& operator-=(const Matrix& rhs);
    Matrix operator*(const Matrix& rhs);
    Matrix& operator*=(const Matrix& rhs);
    Matrix transpose();


    // Access the individual elements
    double& operator()(const unsigned& row, const unsigned& col);
    const double& operator()(const unsigned& row, const unsigned& col) const;
};
