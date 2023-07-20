#!/bin/bash

source $(dirname $0)/config.sh

docker exec -it ${CONTAINER_NAME} zsh