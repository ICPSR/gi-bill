#!/bin/bash

cd /home/slafia/layout-parser/layout-model-training/tools

python train_net.py \
    --dataset_name          card-item \
    --json_annotation_train ../data/cards-v4/train.json \
    --image_path_train      ../data/cards-v4/ \
    --json_annotation_val   ../data/cards-v4/test.json \
    --image_path_val        ../data/cards-v4/ \
    --config-file           ../configs/prima/fast_rcnn_R_50_FPN_3x.yaml \
    OUTPUT_DIR  ../outputs/cards-v4/fast_rcnn_R_50_FPN_3x/ \
    SOLVER.IMS_PER_BATCH 2 
