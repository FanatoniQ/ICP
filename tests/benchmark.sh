#!/bin/sh

GREEN="\033[1;92m"
RED="\033[1;31m"
YELLOW="\033[1;33m"
ANNONCE="\033[38;5;37m"
DEFAULT="\033[1;39m"
RESET="\033[0;39m"

NCU="/opt/cuda/nsight_compute/ncu"
NSIGHT_CLI='/opt/cuda/nsight_compute/nv-nsight-cu-cli'

FILE1="$2"
FILE2="$3"
NB_ITERS="$4"

# ICP
if [ "$1" = "icp" ]; then
    nvprof ./GPUICP $FILE1 $FILE2 $NB_ITERS
fi

if [ "$1" = "icp-batch" ]; then
    nvprof ./GPUICP $FILE1 $FILE2 $NB_ITERS -batch
fi

# Metrics
if [ "$1" = "metric" ]; then
    #nvprof --events all --metrics all ./testgpumean ../data/30points_notebook_P.txt trees 0
    nvprof --metrics all ./GPUICP $FILE1 $FILE2 $NB_ITERS
fi

if [ "$1" = "metric-batch" ]; then
    #nvprof --events all --metrics all ./testgpumean ../data/30points_notebook_P.txt trees 0
    nvprof --metrics all ./GPUICP $FILE1 $FILE2 $NB_ITERS -batch
fi
