#!/bin/bash

source $(dirname $0)/config.sh

# stop the container
# docker stop ${CONTAINER_DISPLAY_NAME}
docker kill ${CONTAINER_DISPLAY_NAME}
docker rm -f ${CONTAINER_DISPLAY_NAME}