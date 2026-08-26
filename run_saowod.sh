#!/bin/bash
set -euo pipefail  # Exit immediately if any command fails, if any variable is undefined, or if a pipeline fails

PORT=${PORT:-"50211"}
GPUS=${GPUS:-"4,5,6,7"}
NUM_GPUS=${NUM_GPUS:-"4"}
SETTING=${SETTING:-"easy"}

# T1
CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py \
                              --num-gpus ${NUM_GPUS} \
                              --dist-url tcp://127.0.0.1:${PORT} \
                              --task t1 \
                              --config-file configs/t1.yaml \
                              OUTPUT_DIR output/t1/ \
                              DATASETS.SETTING ${SETTING}

CUDA_VISIBLE_DEVICES=${GPUS} python discover_unknown.py \
                              --config-file configs/t1_ft.yaml \
                              --input-txt datasets/ImageSets/Main/t1.txt \
                              --task t1 \
                              --output output/t1/unknown_rois.json \
                              --opts MODEL.WEIGHTS output/t1/model_0014999.pth DISCOVER_UNKNOWN True DATASETS.SETTING ${SETTING}

CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py \
                              --num-gpus ${NUM_GPUS} \
                              --dist-url tcp://127.0.0.1:${PORT} \
                              --task t1 \
                              --config-file configs/t1_ft.yaml \
                              --resume \
                              MODEL.WEIGHTS output/t1/model_0014999.pth \
                              DISCOVER_STORE_PATH output/t1/unknown_rois.json \
                              OUTPUT_DIR output/t1/ \
                              DATASETS.SETTING ${SETTING}

# T2
CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py \
                              --num-gpus ${NUM_GPUS} \
                              --dist-url tcp://127.0.0.1:${PORT} \
                              --task t2 \
                              --config-file configs/t2.yaml \
                              MODEL.WEIGHTS output/t1/model_0029999.pth \
                              OUTPUT_DIR output/t2/ \
                              DATASETS.SETTING ${SETTING}

CUDA_VISIBLE_DEVICES=${GPUS} python discover_unknown.py \
                              --config-file configs/t2_ft.yaml \
                              --input-txt datasets/ImageSets/Main/t2_ft.txt \
                              --task t2_ft \
                              --output output/t2/unknown_rois.json \
                              --opts MODEL.WEIGHTS output/t2/model_0014999.pth DISCOVER_UNKNOWN True DATASETS.SETTING ${SETTING}

CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py \
                              --num-gpus ${NUM_GPUS} \
                              --dist-url tcp://127.0.0.1:${PORT} \
                              --task t2_ft \
                              --config-file configs/t2_ft.yaml \
                              --resume \
                              MODEL.WEIGHTS output/t2/model_0014999.pth \
                              DISCOVER_STORE_PATH output/t2/unknown_rois.json \
                              OUTPUT_DIR output/t2/ \
                              DATASETS.SETTING ${SETTING}

# T3
CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py \
                              --num-gpus ${NUM_GPUS} \
                              --dist-url tcp://127.0.0.1:${PORT} \
                              --task t3 \
                              --config-file configs/t3.yaml \
                              MODEL.WEIGHTS output/t2/model_0029999.pth \
                              OUTPUT_DIR output/t3/ \
                              DATASETS.SETTING ${SETTING}

CUDA_VISIBLE_DEVICES=${GPUS} python discover_unknown.py \
                              --config-file configs/t3_ft.yaml \
                              --input-txt datasets/ImageSets/Main/t3_ft.txt \
                              --task t3_ft \
                              --output output/t3/unknown_rois.json \
                              --opts MODEL.WEIGHTS output/t3/model_0014999.pth DISCOVER_UNKNOWN True DATASETS.SETTING ${SETTING}

CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py \
                              --num-gpus ${NUM_GPUS} \
                              --dist-url tcp://127.0.0.1:${PORT} \
                              --task t3_ft \
                              --config-file configs/t3_ft.yaml \
                              --resume \
                              MODEL.WEIGHTS output/t3/model_0014999.pth \
                              DISCOVER_STORE_PATH output/t3/unknown_rois.json \
                              OUTPUT_DIR output/t3/ \
                              DATASETS.SETTING ${SETTING}

# T4
CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py \
                              --num-gpus ${NUM_GPUS} \
                              --dist-url tcp://127.0.0.1:${PORT} \
                              --task t4 \
                              --config-file configs/t4.yaml \
                              MODEL.WEIGHTS output/t3/model_0029999.pth \
                              OUTPUT_DIR output/t4/ \
                              DATASETS.SETTING ${SETTING}

CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py \
                              --num-gpus ${NUM_GPUS} \
                              --dist-url tcp://127.0.0.1:${PORT} \
                              --task t4_ft \
                              --config-file configs/t4_ft.yaml \
                              --resume \
                              MODEL.WEIGHTS output/t4/model_0014999.pth \
                              OUTPUT_DIR output/t4/ \
                              DATASETS.SETTING ${SETTING}