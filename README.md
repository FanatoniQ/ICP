# ICP
CPU (C++) &amp; GPU (CUDA) Iterative closest point implementation


## Installation

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

```bash
./CPUICP ../data/30points_notebook_P.txt ../data/30points_notebook_Q.txt
./GPUICP ../data/30points_notebook_P.txt ../data/30points_notebook_Q.txt
```