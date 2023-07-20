#!/bin/bash

DOCKER_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJ_ROOT=$(dirname "$DOCKER_DIR")
CONTAINER_NAME="bundlesdf"
USER_NAME="user"
USER_ID=$(id -u)
GROUP_ID=$(id -g)
CONDA_ENV_NAME="py39"
CONDA_ENV_PATHON_VERSION="3.9"