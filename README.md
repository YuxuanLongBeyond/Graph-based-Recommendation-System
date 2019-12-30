# Graph-based-Recommendation-System
Graph convolutional matrix completion

## RUN
python3 run.py --train_flag True \
                --test_flag True \
                --rate_num 5 \
                --use_side_feature False \
                --lr 0.01 \
                --weight_decay 0.00001 \
                --num_epochs 1000 \
                --hidden_dim 5 \
                --side_hidden_dim 5 \
                --out_dim 5 \
                --drop_out 0.0 \
                --split_ratio 0.8 \
                --save_steps 100 \
                --verbal_steps 100
