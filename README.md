# ICP
CPU (C++) &amp; GPU (CUDA) Iterative closest point implementation

[![Build + Tests (master)](https://github.com/FanatoniQ/ICP/workflows/CMake/badge.svg)](https://github.com/FanatoniQ/ICP/actions)

## Authors
EPITA SCIA 2021 major GPGPU course project, made by :
* geoffrey.bossut (geoffrey.bossut@epita.fr)
* guillaume.blassel (guillaume.blassel@epita.fr)
* tom.mechineau (tom.mechineau@epita.fr)
* sabrina.meng (sabrina.meng@epita.fr)

## Installation

```bash
mkdir build
cd build
cat ../requirements.system | xargs sudo apt-get install
pip3 install -r ../requirements.txt
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make
```

## Usage

```bash
./CPUICP ../data/30points_notebook_P.txt ../data/30points_notebook_Q.txt 5
./GPUICP ../data/30points_notebook_P.txt ../data/30points_notebook_Q.txt 5
```

## Options
```bash
./CPUICP ${FILE1} ${FILE2} ${NB_ITERS}
```

```bash
./GPUICP ${FILE1} ${FILE2} ${NB_ITERS} (-bash)
```
Option "-bash": Runs the GPU bash implementation

## Testing

### liblinalg (CPU)

numpy + pandas testing for our linear algebra library

Testing mean, dotproduct and svd: (from the build folder)

```bash
python3 ../tests/test.py
```

## Benchmark
```bash
cd build/
cp ../tests/benchmark.sh .
./benchmark icp
```