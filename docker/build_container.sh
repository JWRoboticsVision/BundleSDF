#!/bin/bash

# get the project root directory
CURR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# build the docker image
docker build \
    --network host \
    -t bundlesdf:latest \
    -f $CURR_DIR/dockerfile \
    $CURR_DIR
