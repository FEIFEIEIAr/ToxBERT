#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

for mask in 0.4 0.5
do
    for n_layers in 8 10 12 16
    do
        for embedding_dim in 128 256
        do
            # for data_set in {5..1}
            for data_set in 'docs/DIQTA'
            do
                python train.py \
                    --device cuda \
                    --batch_size 64  \
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
                    --measure_name 'label' \
                    --dataset_name "diqt${data_set}" \
                    --data_root ./data/diqt/${data_set} \
                    --checkpoints_folder './checkpoints_diqt'_${mask} \
                    --mask_prob $mask \
                    --threshold $mask \
                    --mask 'token' \
                # python train.py \
                #     --device cuda \
                #     --batch_size 128  \
                #     --n_heads 8 \
                #     --n_layers $n_layers \
                #     --embedding_dim $embedding_dim \
                #     --en_dropout 0.0 \
                #     --dropout 0.0 \
                #     --lr 1e-3 \
                #     --weight_decay 1e-5 \
                #     --num_workers 14 \
                #     --max_epochs 400 \
                #     --valid_every 1 \
                #     --measure_name DILI_label \
                #     --dataset_name "dili${data_set}" \
                #     --data_root ./data/dili/${data_set} \
                #     --checkpoints_folder './checkpoints_dili'_${mask}_${n_layers} \
                #     --mask_prob $mask \
                #     --threshold $mask \
                #     --mask 'token' \

                # python train.py \
                #     --device cuda \
                #     --batch_size 64  \
                #     --n_heads 8 \
                #     --n_layers 8 \
                #     --embedding_dim $embedding_dim \
                #     --en_dropout 0.0 \
                #     --dropout 0.0 \
                #     --lr 1e-3 \
                #     --weight_decay 1e-5 \
                #     --num_workers 14 \
                #     --max_epochs 400 \
                #     --valid_every 1 \
                #     --measure_name label \
                #     --dataset_name "dira${data_set}" \
                #     --data_root ./data/dira/${data_set} \
                #     --checkpoints_folder './checkpoints_dira'_${mask} \
                #     --mask_prob $mask \
                #     --threshold $mask \
                #     --mask 'token' \
                    
            done
        done
    done
done