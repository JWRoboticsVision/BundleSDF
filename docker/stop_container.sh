#!/bin/bash

source $(dirname $0)/config.sh

# stop the container
# docker stop ${CONTAINER_NAME}
docker kill ${CONTAINER_NAME}
docker rm -f ${CONTAINER_NAME}