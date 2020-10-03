#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

double *readCSV(std::istream &f, std::string &h, size_t *nbaxis, size_t *nbpoints);