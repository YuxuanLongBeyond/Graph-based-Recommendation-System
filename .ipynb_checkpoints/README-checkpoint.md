# Graph-based-Recommendation-System
Graph convolutional matrix completion

## RUN
!python3 run.py --train_flag 1 \
                --test_flag 1 \
                --rate_num 5 \
                --use_side_feature 0 \
                --lr 0.01 \
                --weight_decay 0.00001 \
                --num_epochs 1000 \
                --hidden_dim 5 \
                --side_hidden_dim 5 \
                --out_dim 5 \
                --drop_out 0.0 \
                --split_ratio 0.8 \
                --save_steps 100 \
                --verbal_steps 100 \
                --log_dir './log' \
                --saved_model_folder './parameters' \
                --use_data_whitening 0 

1: 整理代码，数据和训练分隔开，所有function需要有注释
2：README，必要的安装包
3：tutorial notebook