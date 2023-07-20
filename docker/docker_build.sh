#!/bin/bash

source $(dirname $0)/config.sh

# build the docker image
time docker build \
    --network host \
    --build-arg USERNAME=$USER_NAME \
    --build-arg UID=$USER_ID \
    --build-arg GID=$GROUP_ID \
    --build-arg CONDA_ENV_NAME=$CONDA_ENV_NAME \
    --build-arg CONDA_ENV_PATHON_VERSION=$CONDA_ENV_PATHON_VERSION \
    --file ${DOCKER_DIR}/dockerfile \
    --tag ${CONTAINER_NAME}:latest \
    ${DOCKER_DIR}
