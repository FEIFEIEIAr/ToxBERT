#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Mask Ratio
for mask in 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9
do

    python train.py \
        --device cuda \
        --batch_size 64  \
        --n_heads 8 \
        --n_layers 8 \
        --embedding_dim 128 \
        --en_dropout 0.0 \
        --dropout 0.0 \
        --lr 1e-3 \
        --weight_decay 1e-5 \
        --num_workers 14 \
        --max_epochs 400 \
        --valid_every 1 \
        --measure_name label \
        --dataset_name diqt_$mask \
        --data_root docs/DIQTA \
        --checkpoints_folder './checkpoints_diqt_mask' \
        --mask_prob $mask \
        --mask 'token' \
        --threshold 0.5 \

done

# Dropout Ratio
for dropout in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do

    python train.py \
        --device cuda \
        --batch_size 64  \
        --n_heads 8 \
        --n_layers 8 \
        --embedding_dim 128 \
        --en_dropout $dropout \
        --dropout $dropout \
        --lr 1e-3 \
        --weight_decay 1e-5 \
        --num_workers 14 \
        --max_epochs 400 \
        --valid_every 1 \
        --measure_name label \
        --dataset_name diqt_$dropout \
        --data_root docs/DIQTA \
        --checkpoints_folder './checkpoints_diqt_dropout' \
        --mask_prob 0.5 \
        --mask 'token' \
        --threshold 0.5 \

done

# Norm
# Replace norm to identity
for norm in 0 0.2 0.4 0.6 0.8 1
do
    python train_no_norm.py \
        --device cuda \
        --batch_size 64  \
        --n_heads 8 \
        --n_layers 8 \
        --embedding_dim 128 \
        --en_dropout 0.0 \
        --dropout 0.0 \
        --lr 1e-3 \
        --weight_decay 1e-5 \
        --num_workers 14 \
        --max_epochs 400 \
        --valid_every 1 \
        --measure_name label \
        --dataset_name diqt_no_norm_$norm \
        --data_root docs/DIQTA \
        --checkpoints_folder './checkpoints_diqt_no_norm' \
        --mask_prob 0.5 \
        --mask 'token' \
        --drop_norm $norm \

done
# L2
for weight_decay in 0.0 1e-1 1e-2 1e-3 5e-3 1e-4 5e-4 1e-5 5e-4 1e-6
do
    python train.py \
        --device cuda \
        --batch_size 64  \
        --n_heads 8 \
        --n_layers 8 \
        --embedding_dim 128 \
        --en_dropout 0.0 \
        --dropout 0.0 \
        --lr 1e-3 \
        --weight_decay $weight_decay \
        --num_workers 14 \
        --max_epochs 400 \
        --valid_every 1 \
        --measure_name label \
        --dataset_name diqt_$weight_decay \
        --data_root docs/DIQTA \
        --checkpoints_folder './checkpoints_diqt_l2' \
        --mask_prob 0.5 \
        --mask 'token' \

done

# Dim
for embedding_dim in 16 32 64 128 256
do
    python train.py \
        --device cuda \
        --batch_size 64 \
        --n_heads 8 \
        --n_layers 8 \
        --embedding_dim $embedding_dim \
        --en_dropout 0.0 \
        --dropout 0.0 \
        --lr 1e-3 \
        --weight_decay 1e-5 \
        --num_workers 14 \
        --max_epochs 400 \
        --valid_every 1 \
        --measure_name label \
        --dataset_name diqt_$embedding_dim \
        --data_root docs/DIQTA \
        --checkpoints_folder './checkpoints_diqt_dim' \
        --mask_prob 0.5 \
        --mask 'token' \

done

for embedding_dim in 512 768 1024
do
    python train.py \
        --device cuda \
        --batch_size 32 \
        --n_heads 8 \
        --n_layers 8 \
        --embedding_dim $embedding_dim \
        --en_dropout 0.0 \
        --dropout 0.0 \
        --lr 1e-3 \
        --weight_decay 1e-5 \
        --num_workers 14 \
        --max_epochs 400 \
        --valid_every 1 \
        --measure_name label \
        --dataset_name diqt_$embedding_dim \
        --data_root docs/DIQTA \
        --checkpoints_folder './checkpoints_diqt_dim' \
        --mask_prob 0.5 \
        --mask 'token' \
        --accumulate_grad_batches 32 \

done

# head
for head in 1 2 4 8 16 32 64 128
do
    python train.py \
        --device cuda \
        --batch_size 64 \
        --n_heads $head \
        --n_layers 8 \
        --embedding_dim 128 \
        --en_dropout 0.0 \
        --dropout 0.0 \
        --lr 1e-3 \
        --weight_decay 1e-5 \
        --num_workers 14 \
        --max_epochs 400 \
        --valid_every 1 \
        --measure_name label \
        --dataset_name diqt_$head \
        --data_root docs/DIQTA \
        --checkpoints_folder './checkpoints_diqt_head' \
        --mask_prob 0.5 \
        --mask 'token' \
        
done

# Layer
for layer in 2 4 6 8 10 12 14 16
do
    python train.py \
        --device cuda \
        --batch_size 64  \
        --n_heads 8 \
        --n_layers $layer \
        --embedding_dim 128 \
        --en_dropout 0.0 \
        --dropout 0.0 \
        --lr 1e-3 \
        --weight_decay 1e-5 \
        --num_workers 14 \
        --max_epochs 400 \
        --valid_every 1 \
        --measure_name label \
        --dataset_name diqt_$layer \
        --data_root docs/DIQTA \
        --checkpoints_folder './checkpoints_diqt_layer' \
        --mask_prob 0.5 \
        --mask 'token' \

done

for layer in 18 20 22 24 26 28 30 32
do
    python train.py \
        --device cuda \
        --batch_size 32  \
        --n_heads 8 \
        --n_layers $layer \
        --embedding_dim 128 \
        --en_dropout 0.0 \
        --dropout 0.0 \
        --lr 1e-3 \
        --weight_decay 1e-5 \
        --num_workers 14 \
        --max_epochs 400 \
        --valid_every 1 \
        --measure_name label \
        --dataset_name diqt_$layer \
        --data_root docs/DIQTA \
        --checkpoints_folder './checkpoints_diqt_layer' \
        --mask_prob 0.5 \
        --mask 'token' \
        --accumulate_grad_batches 32 \

done