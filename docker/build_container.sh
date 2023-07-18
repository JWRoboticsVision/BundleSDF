#!/bin/bash

source $(dirname $0)/config.sh

# build the docker image
docker build \
    --network host \
    -t ${CONTAINER_NAME}:latest \
    -f ${DOCKER_DIR}/dockerfile \
    ${DOCKER_DIR}
