#include <stdio.h>

#include <cstring>

#include <string>
#include <vector>
//#include <charconv>

#include "error.hpp"
#include "libCSV/csv.hpp"

/**
 ** \brief __readCSV vector method to return the non transposed double array parsed from
 ** the given CSV file
 **
 ** \deprecated we should use strtod or std::from_chars instead of a stringstream
 **
 ** \param f csv input stream
 ** \param h the header string or empty string if no header
 **
 ** \return res the parsed vector of vector
 **/
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

/**
 ** \brief __readCSV vector method to return the non transposed double array parsed from
 ** the given CSV file
 **
 ** \deprecated uses C++ std::vector, which is not consistent with our need for C-style
 ** arrays, at least if we keep this for simplicity sake, do not use stringstream in
 ** called function readCSVLine
 **
 ** \param f csv input stream
 ** \param h the header string or empty string if no header
 **
 ** \return res the parsed vector of vector
 **/
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

double *readCSVT(std::istream &f, std::string &h, size_t *nbaxis, size_t *nbpoints)
{
    double *m;
    size_t i, j = 0;
    auto f1M = __readCSV<double>(f, h);

    std::cerr << h << std::endl;
    *nbaxis = f1M.at(0).size();
    *nbpoints = f1M.size();
    //if (!(m = (double *)calloc(*nbaxis * *nbpoints, sizeof(double))))
    //    errx(2, "alloc error !");
    m = (double *)calloc(*nbaxis * *nbpoints, sizeof(double));
    runtime_assert(m != nullptr, "Alloc error !");
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

double *readCSV(const char *path, std::string &h, size_t &nblines, size_t &nbcols)
{
    FILE *f = fopen(path, "r");
    //if (!f)
    //    errx(1, "File %s could not be opened !", path);
    runtime_assert(f != nullptr, "File could not be opened !");
    ssize_t read;
    size_t len = 0, cols_num;
    char *line = NULL;
    char *token, *end_tok;

    size_t size = 10;
    size_t count = 0;
    double *r = (double *)malloc(size * sizeof(double));
    //if (!r)
    //    errx(1, "Alloc error !");
    runtime_assert(r != nullptr, "Alloc error !");
    nblines = 0;
    nbcols = 0;
    h = "";
    while ((read = getline(&line, &len, f)) != -1)
    {
        if (h.empty())
            h = std::string{line};
        std::cerr << "Line " << nblines << " read: " << line << std::endl;
        cols_num = 0;
        token = strtok(line, ",");
        while (token != NULL)
        {
            cols_num++;
            // add token to the array
            if (count == size)
            {
                size *= 2;
                r = (double *)realloc(r, size * sizeof(double));
                //if (!r)
                //    errx(1, "Alloc error (realloc) !");
                runtime_assert(r != nullptr, "Alloc error (realloc) !");
            }
            double v;
            /**
             ** this does not compile, it should in c++17...
            auto [p, ec] = std::from_chars((const char *)token, (const char *)token + strlen(token), v);
            if (p == token || ec == std::errc())
                errx(2, "Not a valid double !");
            **/
            v = strtod(token, &end_tok);
            if (end_tok == token)
            {
                //if (nblines != 0)
                //errx(2, "Not a valid double !");
                runtime_assert(nblines == 0, "Not a valid double !");
                // this is the header line
            }
            else
            {
                if (nblines == 0)
                    h = "";
                r[count] = v;
                count++;
            }
            token = strtok(NULL, ",\n");
        }
        if (nblines == 0)
            nbcols = cols_num;
        nblines++;
    }
    if (!h.empty()) // header case
        nblines--;
    if (line)
        free(line);
    if (count < size) // removing extra memory
    {
        size = count;
        r = (double *)realloc(r, size * sizeof(double));
        //if (!r)
        //    errx(1, "Alloc error (realloc) !");
        runtime_assert(r != nullptr, "Alloc error (realloc) !");
    }
    fclose(f);
    return r;
}