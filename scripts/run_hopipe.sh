#!/bin/bash

# get the project root directory
CURR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJ_ROOT=$(dirname "$CURR_DIR")
GPU_ID=1
SCRIPT_FILE="run_hopipe.py"

DATASET_DIR=${PROJ_ROOT}/datasets/hopip_data/object_pose_test

ALL_SEQUENCES=(
    "20230221_093329_jikai_right_hammer"
    "20230228_094753_jikai_right_box"
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

for SEQUENCE in ${ALL_SEQUENCES[@]} ; do

    for CAM in ${RS_CAMS[@]} ; do
        INPUT_DIR=${DATASET_DIR}/${SEQUENCE}/${CAM}
        OUTPUT_DIR=${DATASET_DIR}/${SEQUENCE}/data_processing/bundlesdf/${CAM}

        # Run joint tracking and reconstruction
        echo "###############################################################################"
        echo "1. Running joint tracking and reconstruction..."
        echo "###############################################################################"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT_FILE} \
            --mode run_video \
            --video_dir ${INPUT_DIR} \
            --out_folder ${OUTPUT_DIR} \
            --use_segmenter 0 \
            --use_gui 1 \
            --debug_level 2

        # Run global refinement post-processing to refine the mesh
        echo "###############################################################################"
        echo "2. Running global refinement post-processing..."
        echo "###############################################################################"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT_FILE} \
            --mode global_refine \
            --video_dir ${INPUT_DIR} \
            --out_folder ${OUTPUT_DIR}

        # Get the auto-cleaned mesh
        echo "###############################################################################"
        echo "3. Getting the auto-cleaned mesh..."
        echo "###############################################################################"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT_FILE} \
            --mode get_mesh \
            --video_dir ${INPUT_DIR} \
            --out_folder ${OUTPUT_DIR}

    done

done

