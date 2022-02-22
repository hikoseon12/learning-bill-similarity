#!/bin/bash

python train.py \
    --train_data_name human_annotated \
    --train_data_size 3305 \
    --pretrained_model roberta-base \
    --lr 2e-5 \
    --batch_size 32 \
    --num_epochs 4 \
    --gpu_list [0,1,2,3] \
    --nth_result 0 \
    --memo demo 