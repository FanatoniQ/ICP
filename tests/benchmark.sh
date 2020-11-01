#!/bin/sh

GREEN="\033[1;92m"
RED="\033[1;31m"
YELLOW="\033[1;33m"
ANNONCE="\033[38;5;37m"
DEFAULT="\033[1;39m"
RESET="\033[0;39m"

NCU="/opt/cuda/nsight_compute/ncu"
NCU_OPT="--target-processes all"
NSIGHT_CLI='/opt/cuda/nsight_compute/nv-nsight-cu-cli'

# Profiling Summary
if [ $# -eq 0 ]; then
    nvprof ./testgpumean ../data/30points_notebook_P.txt trees 0
fi

# Metrics
if [ "$1" = "metric" ]; then
    #nvprof --events all --metrics all ./testgpumean ../data/30points_notebook_P.txt trees 0
    nvprof --metrics all ./testgpumean ../data/30points_notebook_P.txt trees 0
fi

# NSIGHT Commands
if [ "$1" = "nsight" ]; then
    $NCU $NCU_OPT ./testgpumean ../data/30points_notebook_P.txt dummy 0
    #$NSIGHT_CLI ./GPUICP ../data/30points_notebook_P.txt ../data/30points_notebook_Q.txt
fi
