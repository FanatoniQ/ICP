#include "libalg/print.hpp"

void print_matrix(std::ostream &s, float *a, int m, int n)
{
    print_matrix(s, a, m, n, m);
}

void print_matrix(std::ostream &s, float *a, int m, int n, int lda)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++)
        {
            s << a[j + i * lda]; // printf "%6.15f",
            if (j != m - 1)
                s << ",";
        }
        s << std::endl;
    }
    s << std::endl
      << std::endl;
}