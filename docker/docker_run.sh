#!/bin/bash

source $(dirname $0)/config.sh

# give permissions to the Docker client to connect to your X server
xhost +local:${USER_NAME}

docker run \
    --detach \
    --tty \
    --gpus 'all,"capabilities=compute,utility,graphics"' \
    --ipc=host \
    --ulimit memlock=-1 \
    --network=host \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    --env DISPLAY=${DISPLAY} \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --volume "${PROJ_ROOT}":/home/${USER_NAME}/workspace \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume /etc/localtime:/etc/localtime:ro \
    --name ${CONTAINER_NAME} \
    ${CONTAINER_NAME}:latest

docker ps -a

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