#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
nproc_per_node=2
batch_size=16
max_iterations=40000   #40000
save_per_iters=1600  #1600

# Repeat 3 times
# train_baseline
# labels=(1 2 3)
labels=(1)
for label in "${labels[@]}"
do
    python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} \
        train_student.py \
        --teacher-model deeplabv3 \
        --teacher-backbone resnet18 \
        --data potsdam256_stride \
        --batch-size ${batch_size} \
        --max-iterations ${max_iterations} \
        --save-per-iters ${save_per_iters} \
        --val-per-iters ${save_per_iters} \
        --save-dir " " \
        --log-dir " " \
        --teacher-pretrained-base  pretrain/resnet18-imagenet.pth
done

