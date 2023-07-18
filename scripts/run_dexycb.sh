#!/bin/bash

# get the project root directory
CURR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJ_ROOT=$(dirname "$CURR_DIR")
SCRIPT_FILE="run_dexycb.py"
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

    rm -rf ${DATASET_DIR}/${SEQUENCE}/data_processing/bundlesdf

    for CAM in ${RS_CAMS[@]} ; do
        INPUT_DIR=${DATASET_DIR}/${SEQUENCE}/${CAM}
        OUTPUT_DIR=${DATASET_DIR}/${SEQUENCE}/data_processing/bundlesdf/${CAM}

        echo "###############################################################################"
        echo "# Running demo on ${INPUT_DIR}..."
        echo "###############################################################################"

        # Run joint tracking and reconstruction
        echo "==============================================================================="
        echo "==============1. Running joint tracking and reconstruction...=================="
        echo "==============================================================================="
        CUDA_VISIBLE_DEVICES=$GPU_ID python ${SCRIPT_FILE} \
            --mode run_video \
            --video_dir ${INPUT_DIR} \
            --out_folder ${OUTPUT_DIR} \
            --use_segmenter 0 \
            --use_gui 1 \
            --debug_level 2

        # Run global refinement post-processing to refine the mesh
        echo "==============================================================================="
        echo "==============2. Running global refinement post-processing...=================="
        echo "==============================================================================="
        CUDA_VISIBLE_DEVICES=$GPU_ID python ${SCRIPT_FILE} \
            --mode global_refine \
            --video_dir ${INPUT_DIR} \
            --out_folder ${OUTPUT_DIR}

        # Get the auto-cleaned mesh
        echo "==============================================================================="
        echo "==============3. Getting the auto-cleaned mesh...=============================="
        echo "==============================================================================="
        CUDA_VISIBLE_DEVICES=$GPU_ID python ${SCRIPT_FILE} \
            --mode get_mesh \
            --video_dir ${INPUT_DIR} \
            --out_folder ${OUTPUT_DIR}

    done

done
