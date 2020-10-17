#pragma once

#include <string>
#include <stdexcept>

#define runtime_failure(msg)           \
    {                                  \
        throw std::runtime_error(msg); \
    }

#define runtime_assert(cond, msg)          \
    {                                      \
        if (!(cond))                       \
            throw std::runtime_error(msg); \
    }
