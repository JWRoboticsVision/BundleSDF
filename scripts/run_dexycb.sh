#!/bin/bash

# get the project root directory
CURR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJ_ROOT=$(dirname "$CURR_DIR")
GPU_ID=0
SCRIPT_FILE="run_dexycb.py"

DATASET_DIR=${PROJ_ROOT}/datasets/dex_ycb_selected


ALL_SEQUENCES=(
    "20200709-subject-01/20200709_142321"
    "20200908-subject-05/20200908_143714"
    "20200918-subject-06/20200918_113049"
    "20200918-subject-06/20200918_113532"
    "20200918-subject-06/20200918_113738"
    "20200918-subject-06/20200918_113936"
    "20200918-subject-06/20200918_114208"
    "20200918-subject-06/20200918_114409"
    "20200918-subject-06/20200918_114619"
    "20200918-subject-06/20200918_114840"
    "20200918-subject-06/20200918_115049"
    "20200918-subject-06/20200918_115312"
    "20200918-subject-06/20200918_115540"
    "20200918-subject-06/20200918_115755"
    "20200918-subject-06/20200918_120016"
    "20200918-subject-06/20200918_120215"
    "20200918-subject-06/20200918_120425"
    "20200918-subject-06/20200918_120623"
    "20200918-subject-06/20200918_120838"
    "20200918-subject-06/20200918_121101"
    "20200918-subject-06/20200918_121314"
    "20200928-subject-07/20200928_143944"
    "20201002-subject-08/20201002_104620"
    "20201002-subject-08/20201002_105313"
    "20201002-subject-08/20201002_105343"
    "20201002-subject-08/20201002_105558"
    "20201002-subject-08/20201002_105803"
    "20201002-subject-08/20201002_110017"
    "20201002-subject-08/20201002_110227"
    "20201002-subject-08/20201002_110453"
    "20201002-subject-08/20201002_110715"
    "20201002-subject-08/20201002_110940"
    "20201002-subject-08/20201002_111215"
    "20201002-subject-08/20201002_111422"
    "20201002-subject-08/20201002_111644"
    "20201002-subject-08/20201002_111900"
    "20201002-subject-08/20201002_112114"
    "20201002-subject-08/20201002_112335"
    "20201002-subject-08/20201002_112558"
    "20201002-subject-08/20201002_112816"
    "20201002-subject-08/20201002_113031"
    "20201015-subject-09/20201015_142601"
    "20201015-subject-09/20201015_143403"
    "20201015-subject-09/20201015_143636"
    "20201015-subject-09/20201015_143857"
    "20201015-subject-09/20201015_144145"
    "20201015-subject-09/20201015_144414"
    "20201015-subject-09/20201015_144721"
    "20201015-subject-09/20201015_145003"
    "20201015-subject-09/20201015_145240"
    "20201015-subject-09/20201015_145515"
    "20201015-subject-09/20201015_145737"
    "20201015-subject-09/20201015_145956"
    "20201015-subject-09/20201015_150208"
    "20201015-subject-09/20201015_150441"
    "20201015-subject-09/20201015_150710"
    "20201015-subject-09/20201015_150938"
    "20201015-subject-09/20201015_151221"
    "20201015-subject-09/20201015_151450"
    "20201022-subject-10/20201022_111123"
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

for SEQUENCE in ${ALL_SEQUENCES[@]} ; do

    for CAM in ${RS_CAMS[@]} ; do
        INPUT_DIR=${DATASET_DIR}/${SEQUENCE}/${CAM}
        OUTPUT_DIR=${DATASET_DIR}/${SEQUENCE}/data_processing/bundlesdf/${CAM}

        # Run joint tracking and reconstruction
        echo "###############################################################################"
        echo "1. Running joint tracking and reconstruction..."
        echo "###############################################################################"
        CUDA_VISIBLE_DEVICES=$GPU_ID python ${SCRIPT_FILE} \
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
        CUDA_VISIBLE_DEVICES=$GPU_ID python ${SCRIPT_FILE} \
            --mode global_refine \
            --video_dir ${INPUT_DIR} \
            --out_folder ${OUTPUT_DIR}

        # Get the auto-cleaned mesh
        echo "###############################################################################"
        echo "3. Getting the auto-cleaned mesh..."
        echo "###############################################################################"
        CUDA_VISIBLE_DEVICES=$GPU_ID python ${SCRIPT_FILE} \
            --mode get_mesh \
            --video_dir ${INPUT_DIR} \
            --out_folder ${OUTPUT_DIR}

    done

done



# # Run joint tracking and reconstruction
# echo "###############################################################################"
# echo "1. Running joint tracking and reconstruction..."
# echo "###############################################################################"
# python run_dex_ycb.py \
#     --mode run_video \
#     --video_dir ${INPUT_DIR} \
#     --out_folder ${OUTPUT_DIR} \
#     --use_segmenter 0 \
#     --use_gui 1 \
#     --debug_level 2

# # Run global refinement post-processing to refine the mesh
# echo "###############################################################################"
# echo "2. Running global refinement post-processing..."
# echo "###############################################################################"
# python run_dex_ycb.py \
#     --mode global_refine \
#     --video_dir ${INPUT_DIR} \
#     --out_folder ${OUTPUT_DIR}

# # Get the auto-cleaned mesh
# echo "###############################################################################"
# echo "3. Getting the auto-cleaned mesh..."
# echo "###############################################################################"
# python run_dex_ycb.py \
#     --mode get_mesh \
#     --video_dir ${INPUT_DIR} \
#     --out_folder ${OUTPUT_DIR}
