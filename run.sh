#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

for i in $*
do
echo "Will use $i in the script"
python3 run.py --train_flag 1 \
                --test_flag 1 \
                --rate_num 5 \
                --use_side_feature 0 \
                --lr $i \
                --weight_decay 0.00001 \
                --num_epochs 100000 \
                --hidden_dim 5 \
                --side_hidden_dim 5 \
                --out_dim 5 \
                --drop_out 0.0 \
                --split_ratio 0.8 \
                --save_steps 100 \
                --verbal_steps 100 \
                --log_dir './log' \
                --saved_model_folder './parameters' \
                --use_data_whitening 0 \
                --use_laplacian_loss 1 \
                --laplacian_loss_weight 0.1
done