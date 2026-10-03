# We use the Car dataset as an example. You are free to change the dataset name and class number for other datasets.
# export DATASET="car"
# export CLASS_NUMBER=196
# export DATASET="pet" 
# export CLASS_NUMBER=37
# export DATASET="aircraft"
# export CLASS_NUMBER=100
export DATASET="cub"
export CLASS_NUMBER=200

export N_WORKERS=8 # 8 by default, you can change it according to your GPU memory size. For example, if you have 4 GPUs, you can set N_WORKERS=4.


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py  \
    --pretrained \
    --arch="resnet50" \
    --epochs=128 \
    --batch_size=256 \
    --lr=0.01 \
    --size=224 \
    --seed=2020 \
    --syn_p=0 \
    --resize=256 \
    --syn_dir='syn/${DATASET}/5shot/ours_0.3_5.0' \
    --datasets=$DATASET \
    --num_class=$CLASS_NUMBER \
    --use_cutmix \
    --mixup_probability=1.0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="vit_b_16" \
    --epochs=100 \
    --batch_size=32 \
    --lr=1e-3 \
    --size=384 \
    --seed=2020 \
    --syn_p=0 \
    --resize=440 \
    --syn_dir="syn/${DATASET}/5shot/ours_0.3_5.0" \
    --datasets=$DATASET \
    --num_class=$CLASS_NUMBER \
    --use_cutmix \
    --mixup_probability=1.0
