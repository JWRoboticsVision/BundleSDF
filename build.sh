#!/bin/bash

# get the project root directory
PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# build mycuda
echo "###############################################"
echo "Building mycuda..."
echo "###############################################"
cd ${PROJ_ROOT}/mycuda && \
    rm -rf build *egg* *.so && \
    python -m pip install -e .

# build BundleTrack
echo "###############################################"
echo "Building BundleTrack..."
echo "###############################################"
cd ${PROJ_ROOT}/BundleTrack && \
    rm -rf build && \
    mkdir build && \
    cd build && \
    cmake .. -Wno-dev && \
    make -j$(nproc)
