#include "libalg/CPUView.hpp"

CPUView::CPUView(double *line, size_t dim1)
{
    this->array = line;
    this->dim0 = 1;
    this->dim1 = dim1;
}

CPUView::~CPUView()
{
    this->array = nullptr; // avoid invalid free
}
