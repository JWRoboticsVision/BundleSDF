#!/bin/bash

# get the project root directory
CURR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJ_ROOT=$(dirname "$CURR_DIR")

INPUT_DIR=${PROJ_ROOT}/test/demo/2022-11-18-15-10-24_milk
OUTPUT_DIR=${PROJ_ROOT}/test/demo/bundlesdf_2022-11-18-15-10-24_milk

export CUDA_VISIBLE_DEVICES=0

# Run joint tracking and reconstruction
echo "###############################################################################"
echo "1. Running joint tracking and reconstruction..."
echo "###############################################################################"
python run_custom.py \
    --mode run_video \
    --video_dir ${INPUT_DIR} \
    --out_folder ${OUTPUT_DIR} \
    --use_segmenter 1 \
    --use_gui 1 \
    --debug_level 2

# Run global refinement post-processing to refine the mesh
echo "###############################################################################"
echo "2. Running global refinement post-processing..."
echo "###############################################################################"
python run_custom.py \
    --mode global_refine \
    --video_dir ${INPUT_DIR} \
    --out_folder ${OUTPUT_DIR}

# Get the auto-cleaned mesh
echo "###############################################################################"
echo "3. Getting the auto-cleaned mesh..."
echo "###############################################################################"
python run_custom.py \
    --mode get_mesh \
    --video_dir ${INPUT_DIR} \
    --out_folder ${OUTPUT_DIR}
