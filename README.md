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

## Example Usage

```bash
./CPUICP ../data/30points_notebook_P.txt ../data/30points_notebook_Q.txt 5
./GPUICP ../data/30points_notebook_P.txt ../data/30points_notebook_Q.txt 5 -shared-loop
```

## Options
### CPU
```bash
./CPUICP ${FILE1} ${FILE2} ${NB_ITERS}
```
### GPU
Four different implementations at your disposition:
- "-loop"
- "-shared"
- "-shared-loop"
- "-batch"
Example:
```bash
./GPUICP ${FILE1} ${FILE2} ${NB_ITERS} -shared-loop
```

## Testing

### liblinalg (CPU)

numpy + pandas testing for our linear algebra library

Testing mean, dotproduct and svd: (from the build folder)

```bash
python3 ../tests/test.py
```

### Extra tests

Run every test{name} executables in the build folder to get usages and test out the features.

Example:
```bash
./testgpusvd ../data/3ptsP.txt
```

## Benchmark
```bash
cd build/
cp ../tests/benchmark.sh .
```

Default version:
```bash
./benchmark icp ${FILE1} ${FILE2} ${NB_ITERS}
```
Default full metric version :
```bash
./benchmark metric ${FILE1} ${FILE2} ${NB_ITERS}
```

Batch version:
```bash
./benchmark batch ${FILE1} ${FILE2} ${NB_ITERS}
```
Batch full metric version:
```bash
./benchmark metric-batch ${FILE1} ${FILE2} ${NB_ITERS}
```
