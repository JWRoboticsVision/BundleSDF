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
    --volume "${PROJ_ROOT}":/home/${USER_NAME}/${WORK_DIR} \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume /etc/localtime:/etc/localtime:ro \
    --name ${CONTAINER_DISPLAY_NAME} \
    ${CONTAINER_NAME}:${CONTAINER_TAG}

sleep 1
docker ps -a