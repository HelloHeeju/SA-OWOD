#!/bin/bash

PORT=${PORT:-"50211"}
GPUS=${GPUS:-"4,5,6,7"}
NUM_GPUS=${NUM_GPUS:-"4"}

CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py --num-gpus ${NUM_GPUS} --dist-url tcp://127.0.0.1:${PORT} --task t1 --config-file configs/t1.yaml --eval-only MODEL.WEIGHTS output/t1/model_0029999.pth

CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py --num-gpus ${NUM_GPUS} --dist-url tcp://127.0.0.1:${PORT} --task t2_ft --config-file configs/t2_ft.yaml --eval-only MODEL.WEIGHTS output/t2/model_0029999.pth

CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py --num-gpus ${NUM_GPUS} --dist-url tcp://127.0.0.1:${PORT} --task t3_ft --config-file configs/t3_ft.yaml --eval-only MODEL.WEIGHTS output/t3/model_0029999.pth

CUDA_VISIBLE_DEVICES=${GPUS} python train_net.py --num-gpus ${NUM_GPUS} --dist-url tcp://127.0.0.1:${PORT} --task t4_ft --config-file configs/t4_ft.yaml --eval-only MODEL.WEIGHTS output/t4/model_0029999.pth