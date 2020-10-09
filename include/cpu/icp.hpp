#pragma once

#include <vector>
#include <tuple>

#include "libalg/alg.hpp"

std::vector<std::tuple<size_t, int>> get_correspondence_indices(double *P,
    double *Q, size_t P_r, size_t P_c, size_t Q_r, size_t Q_c);
