#!/bin/bash

# get the project root directory
CURR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJ_ROOT=$(dirname "$CURR_DIR")

# give permissions to the Docker client to connect to your X server
xhost +local:docker

# run the container
docker run \
    --gpus all \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    --env DISPLAY=${DISPLAY} \
    -it --rm  \
    --network=host \
    --name bundlesdf \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v "$PROJ_ROOT":/app \
    --ipc=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    bundlesdf:latest \
    zsh