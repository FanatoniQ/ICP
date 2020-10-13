#pragma once

#include <ostream>
#include <vector>

class Matrix {
private:
    double* array;
    size_t dim0;
    size_t dim1;
public:
    // Constructors
    Matrix();
    Matrix(double *array, int dim0, int dim1);
    /**
     * Initialize matrix of dim0*dim1 with init value
     * Throws std::bad_alloc on failed malloc
     * @param dim0
     * @param dim1
     * @param init
     */
    Matrix(size_t dim0, size_t dim1, double init);

    /**
     * Free the ressources
     */
    virtual ~Matrix();

    // Getters and setters
    // TODO remove
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
    std::vector<double> mean_axis(size_t dim);

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

    // Matrix/scalar operations
    Matrix operator+(const double& rhs);
    Matrix operator-(const double& rhs);
    Matrix operator*(const double& rhs);
    Matrix operator/(const double& rhs);

    // Matrix/vector operations
    std::vector<double> operator*(const std::vector<double>& rhs);
    std::vector<double> diag_vec();

    // Access the individual elements
    double& operator()(const unsigned& row, const unsigned& col);
    const double& operator()(const unsigned& row, const unsigned& col) const;

};
