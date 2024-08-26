#!/bin/bash

# other KD methods integrate with DCS
export CUDA_VISIBLE_DEVICES=1,2
nproc_per_node=2
batch_size=16
max_iterations=40000
save_per_iters=1600

repeats=(1 2)

for repeat in "${repeats[@]}"
do

    # train_at.sh
    python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} \
        train_other_kd_integrate.py.py \
        --teacher-model deeplabv3 \
        --student-model deeplabv3 \
        --teacher-backbone resnet101 \
        --student-backbone resnet18 \
        --lambda-kd 1.0 \
        --lambda-at 10000. \
        --data potsdam256_stride \
        --batch-size ${batch_size} \
        --max-iterations ${max_iterations} \
        --save-per-iters ${save_per_iters} \
        --val-per-iters ${save_per_iters} \
        --save-dir "" \
        --log-dir "" \
				--teacher-pretrained kd_deeplabv3_resnet101_vaihingen_best_model.pth  \
        --student-pretrained-base pretrain/resnet18-imagenet.pth

#    # train_dsd.sh
    python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} \
        train_other_kd_integrate.py.py \
        --teacher-model deeplabv3 \
        --student-model deeplabv3 \
        --teacher-backbone resnet101 \
        --student-backbone resnet18 \
        --lambda-psd 1000. \
        --lambda-csd 10. \
        --data potsdam256_stride \
        --batch-size ${batch_size} \
        --max-iterations ${max_iterations} \
        --save-per-iters ${save_per_iters} \
        --val-per-iters ${save_per_iters} \
        --save-dir "" \
        --log-dir "" \
				--teacher-pretrained kd_deeplabv3_resnet101_vaihingen_best_model.pth  \
        --student-pretrained-base pretrain/resnet18-imagenet.pth
#
#
#    # train cirkd.sh
    python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} \
        train_cirkd_integrate.py \
        --teacher-model deeplabv3 \
        --student-model deeplabv3 \
        --teacher-backbone resnet101 \
        --student-backbone resnet18 \
        --data potsdam256_stride \
        --batch-size ${batch_size} \
        --max-iterations ${max_iterations} \
        --save-per-iters ${save_per_iters} \
        --val-per-iters ${save_per_iters} \
        --save-dir "" \
        --log-dir "" \
				--teacher-pretrained kd_deeplabv3_resnet101_vaihingen_best_model.pth  \
        --student-pretrained-base pretrain/resnet18-imagenet.pth
  #cwd
    python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} \
        train_other_kd_integrate.py.py  \
        --teacher-model deeplabv3 \
        --student-model deeplabv3 \
        --teacher-backbone resnet101 \
        --student-backbone resnet18 \
        --lambda-cwd-logit 3. \
        --lambda-cwd-fea 50. \
        --data potsdam256_stride \
        --batch-size ${batch_size} \
        --max-iterations ${max_iterations} \
        --save-per-iters ${save_per_iters} \
        --val-per-iters ${save_per_iters} \
        --save-dir "" \
        --log-dir "" \
				--teacher-pretrained kd_deeplabv3_resnet101_vaihingen_best_model.pth  \
				--student-pretrained-base pretrain/resnet18-imagenet.pth

done