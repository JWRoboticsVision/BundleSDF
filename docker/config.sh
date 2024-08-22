#!/bin/bash

DOCKER_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJ_ROOT=$(dirname "$DOCKER_DIR")

CONTAINER_NAME="bundlesdf/devel"
CONTAINER_DISPLAY_NAME="bundlesdf"
CONTAINER_TAG="latest"

USER_NAME="my_user"
USER_ID=$(id -u)
GROUP_ID=$(id -g)
CONDA_ENV_NAME="bundlesdf"
CONDA_ENV_PATHON_VERSION="3.10"
CUDA_ARCH="6.0 6.1 7.0 7.5 8.0 8.6"
WORK_DIR="code"