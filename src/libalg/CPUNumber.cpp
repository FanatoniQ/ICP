#include "libalg/CPUNumber.hpp"


CPUNumber::CPUNumber(double *value){
    this->array = value;
    this->dim0 = 1;
    this->dim1 = 1;
}
