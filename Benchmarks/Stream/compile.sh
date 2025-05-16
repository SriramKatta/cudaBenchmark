#!/bin/bash -l

module purge
module load cuda cmake


export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
cmake -S . -B build/ -DCMAKE_BUILD_TYPE:STRING=Release  -DDISABLE_NVTX=ON
cmake --build build/ -j 