#!/bin/bash

# get the project root directory
CURR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJ_ROOT=$(dirname "$CURR_DIR")
SCRIPT_FILE=${PROJ_ROOT}/run_hopipe.py
DATASET_DIR=${PROJ_ROOT}/datasets/HoPipe/demo

# Set the GPU ID
if [ -z "$1" ]
then
    GPU_ID=0
else
    GPU_ID=$1
fi

# Set the sequences and cameras
ALL_SEQUENCES=(
    "jikai_right_hammer"
    "jikai_right_power_drill"
)

RS_CAMS=(
    "037522251142"
    "043422252387"
    "046122250168"
    "105322251225"
    "105322251564"
    "108222250342"
    "115422250549"
    "117222250549"
)

# Run the demo
for SEQUENCE in ${ALL_SEQUENCES[@]} ; do

    for CAM in ${RS_CAMS[@]} ; do

        echo "###############################################################################"
        echo "# Processing sequence ${SEQUENCE} - ${CAM}"
        echo "###############################################################################"

        CUDA_VISIBLE_DEVICES=$GPU_ID python ${SCRIPT_FILE} \
            --sequence_folder ${DATASET_DIR}/${SEQUENCE} \
            --serial ${CAM} \
            --use_gui 1 \
            --debug_level 2

    done

done
