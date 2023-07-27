#!/bin/bash

# get the project root directory
CURR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJ_ROOT=$(dirname "$CURR_DIR")
SCRIPT_FILE=${PROJ_ROOT}/run_dexycb.py
DATASET_DIR=${PROJ_ROOT}/datasets/DexYcb/demo

# Set the GPU ID
if [ -z "$1" ]
then
    GPU_ID=0
else
    GPU_ID=$1
fi

# Set the sequences and cameras
ALL_SEQUENCES=(
    "002_master_chef_can"
    "003_cracker_box"
    "011_banana"
    "035_power_drill"
)

RS_CAMS=(
    "836212060125"
    "839512060362"
    "840412060917"
    "841412060263"
    "932122060857"
    "932122060861"
    "932122061900"
    "932122062010"
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
