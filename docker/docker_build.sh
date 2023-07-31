#!/bin/bash

source $(dirname $0)/config.sh

# build the docker image
time docker build \
    --network host \
    --build-arg CUDA_ARCH="${CUDA_ARCH}" \
    --build-arg USERNAME=${USER_NAME} \
    --build-arg UID=${USER_ID} \
    --build-arg GID=${GROUP_ID} \
    --build-arg CONDA_ENV_NAME=${CONDA_ENV_NAME} \
    --build-arg CONDA_ENV_PATHON_VERSION=${CONDA_ENV_PATHON_VERSION} \
    --build-arg EIGEN_VERSION=${EIGEN_VERSION} \
    --build-arg YAML_CPP_VERSION=${YAML_CPP_VERSION} \
    --build-arg PYBIND11_VERSION=${PYBIND11_VERSION} \
    --build-arg OPENCV_VERSION=${OPENCV_VERSION} \
    --build-arg PCL_VERSION=${PCL_VERSION} \
    --build-arg PYTORCH3D_VERSION=${PYTORCH3D_VERSION} \
    --build-arg KAOLIN_VERSION=${KAOLIN_VERSION} \
    --build-arg WORK_DIR=${WORK_DIR} \
    --file ${DOCKER_DIR}/Dockerfile \
    --tag ${CONTAINER_NAME}:${CONTAINER_TAG} \
    ${DOCKER_DIR}
