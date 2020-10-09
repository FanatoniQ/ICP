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

    // Getters and setters
    double *getArray() const;
    void setArray(double *array);
    size_t getDim0() const;
    void setDim0(size_t dim0);
    size_t getDim1() const;
    void setDim1(size_t dim1);

    // Output
    friend std::ostream &operator<<(std::ostream &os, const Matrix &matrix);

    // Operations
    bool operator==(const Matrix &rhs) const;
    bool operator!=(const Matrix &rhs) const;
    std::vector<double> mean_axis(size_t dim)

};
