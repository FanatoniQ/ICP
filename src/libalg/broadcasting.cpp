#include "libalg/broadcasting.hpp"
#include "libalg/alg.hpp"

bool broadcastable(size_t a_0, size_t a_1, size_t b_0, size_t b_1)
{
    if (b_1 != a_1 && b_1 != 1 && a_1 != 1)
        return false;
    if (b_0 != a_0 && b_0 != 1 && a_0 != 1)
        return false;
    return true;
}

bool get_broadcastable_size(size_t a_0, size_t a_1, size_t b_0, size_t b_1, size_t *r_0, size_t *r_1)
{
    if (!broadcastable(a_0, a_1, b_0, b_1))
        return false;
    *r_0 = MAX(a_0, b_0);
    *r_1 = MAX(a_1, b_1);
    return true;
}