#!/bin/bash

# get the project root directory
CURR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJ_ROOT=$(dirname "$CURR_DIR")
SCRIPT_FILE="run_custom.py"
DATASET_DIR=${PROJ_ROOT}/datasets/Demo

# Set the GPU ID
if [ -z "$1" ]
then
    GPU_ID=0
else
    GPU_ID=$1
fi

# Run the demo
INPUT_DIR=${DATASET_DIR}/2022-11-18-15-10-24_milk
OUTPUT_DIR=${DATASET_DIR}/bundlesdf_2022-11-18-15-10-24_milk

rm -rf ${OUTPUT_DIR}

echo "###############################################################################"
echo "# Running demo on ${INPUT_DIR}..."
echo "###############################################################################"

# Run joint tracking and reconstruction
echo "==============================================================================="
echo "==============1. Running joint tracking and reconstruction...=================="
echo "==============================================================================="
CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT_FILE} \
    --mode run_video \
    --video_dir ${INPUT_DIR} \
    --out_folder ${OUTPUT_DIR} \
    --use_segmenter 1 \
    --use_gui 1 \
    --debug_level 2

# Run global refinement post-processing to refine the mesh
echo "==============================================================================="
echo "==============2. Running global refinement post-processing...=================="
echo "==============================================================================="
CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT_FILE} \
    --mode global_refine \
    --video_dir ${INPUT_DIR} \
    --out_folder ${OUTPUT_DIR}

# Get the auto-cleaned mesh
echo "==============================================================================="
echo "==============3. Getting the auto-cleaned mesh...=============================="
echo "==============================================================================="
CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT_FILE} \
    --mode get_mesh \
    --video_dir ${INPUT_DIR} \
    --out_folder ${OUTPUT_DIR}
