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

## Testing

### liblinalg (CPU)

numpy + pandas testing for our linear algebra library

Testing mean and dotproduct: (from the build folder)

```bash
python3 ../tests/test.py ../data/30points_notebook_P.txt mean
python3 ../tests/test.py ../data/30points_notebook_P.txt ../data/30points_notebook_Q.txt dotproduct
```