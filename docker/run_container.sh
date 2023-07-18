#!/bin/bash

source $(dirname $0)/config.sh

# give permissions to the Docker client to connect to your X server
xhost +local:docker

# run the container
# docker run \
#     --gpus all \
#     --env NVIDIA_DISABLE_REQUIRE=1 \
#     --env DISPLAY=${DISPLAY} \
#     -it \
#     --rm  \
#     --network=host \
#     --name ${CONTAINER_NAME} \
#     --cap-add=SYS_PTRACE \
#     --security-opt seccomp=unconfined \
#     -v "${PROJ_ROOT}":/app \
#     --ipc=host \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     ${CONTAINER_NAME}:latest \
#     /bin/zsh

docker run \
    --gpus all \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    --env DISPLAY=${DISPLAY} \
    -dt \
    --network=host \
    --name ${CONTAINER_NAME} \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v "${PROJ_ROOT}":/app \
    --ipc=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    ${CONTAINER_NAME}:latest
