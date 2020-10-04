#include <err.h>

#include <string>
#include <vector>

#include "libCSV/csv.hpp"

template <typename T>
static inline std::vector<T> readCSVLine(std::istringstream &line)
{
    std::vector<T> res;

    while (line.good())
    {
        std::string substr;
        std::getline(line, substr, ',');
        T v;
        std::istringstream ss(substr);
        if (!(ss >> v))
            return std::vector<T>{};
        res.push_back(v);
    }
    return res;
}

template <typename T>
static std::vector<std::vector<T>> __readCSV(std::istream &f, std::string &h)
{
    std::vector<std::vector<T>> res;
    std::string line;
    while (std::getline(f, line))
    {
        std::istringstream iss(line);
        auto values = readCSVLine<T>(iss);
        if (res.empty() && values.empty())
            h = line;
        else
            res.push_back(values);
    }
    return res;
}

// explicit instantiation of template for library export...
//template std::vector<std::vector<double>> readCSV<double>(std::istream &, std::string &);

/**
 *  returns the transposed csv file
 * (each line is constituted of nbaxis points)
 **/
double *readCSV(std::istream &f, std::string &h, size_t *nbaxis, size_t *nbpoints)
{
    double *m;
    size_t i, j = 0;
    auto f1M = __readCSV<double>(f, h);

    std::cerr << h << std::endl;
    *nbaxis = f1M.at(0).size();
    *nbpoints = f1M.size();
    if (!(m = (double *)calloc(*nbaxis * *nbpoints, sizeof(double))))
        errx(2, "alloc error !");
    for (const auto &f1line : f1M)
    {
        i = 0; // i is axis
        for (const auto &f1V : f1line)
        {
            m[i * (*nbpoints) + j] = f1V; // non linear access
            //std::cerr << f1V << "\t";
            ++i;
        }
        //std::cerr << std::endl;
        ++j; // j is line index
    }
    return m;
}