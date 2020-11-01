#!/bin/sh

NCU='/opt/cuda/nsight_compute/ncu'
NCU_OPT='--target-processes all'

NSIGHT_CLI='/opt/cuda/nsight_compute/nv-nsight-cu-cli'

nvprof ./testgpumean ../data/30points_notebook_P.txt trees 0
#$NCU $NCU_OPT ./testgpumean ../data/30points_notebook_P.txt dummy 0
#$NSIGHT_CLI ./GPUICP ../data/30points_notebook_P.txt ../data/30points_notebook_Q.txt
