#include "libalg/CPUNumber.hpp"


CPUNumber::CPUNumber(double value){
    this->value = value;
    this->array = &this->value;
    this->dim0 = 1;
    this->dim1 = 1;
}
