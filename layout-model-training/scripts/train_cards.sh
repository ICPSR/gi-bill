#!/bin/bash

cd /home/slafia/layout-parser/layout-model-training/tools

python train_net.py \
    --dataset_name          card-item \
    --json_annotation_train ../data/cards-v5/train.json \
    --image_path_train      ../data/cards-v5/ \
    --json_annotation_val   ../data/cards-v5/test.json \
    --image_path_val        ../data/cards-v5/ \
    --config-file           ../configs/prima/fast_rcnn_R_50_FPN_3x.yaml \
    OUTPUT_DIR  ../outputs/cards-v5/fast_rcnn_R_50_FPN_3x/ \
    SOLVER.IMS_PER_BATCH 2 
