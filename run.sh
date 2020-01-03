#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

array_lr=()
array_epochs=()
array_hidden_dim=()
array_side_hidden_dim=()
array_dropout=()
array_use_side_feature=()
array_use_data_whitening=()
array_use_laplacian_loss=()
array_laplacian_loss_weight=()
temp=0

echo "here start"
for i in $*
do 
    if [ $i == "-lr" ];then
        temp=1
        continue
    fi
    
    if [ $i == "-epochs" ];then
        temp=2
        continue
    fi
    
    if [ $i == "-hidden_dim" ];then
        temp=3
        continue
    fi
    
    if [ $i == "-side_hidden_dim" ];then
        temp=4
        continue
    fi
    
    if [ $i == "-dropout" ];then
        temp=5
        continue
    fi
    
    if [ $i == "-use_side_feature" ];then
        temp=6
        continue
    fi
    
    if [ $i == "-use_data_whitening" ];then
        temp=7
        continue
    fi
    
    if [ $i == "-use_laplacian_loss" ];then
        temp=8
        continue
    fi
    
    if [ $i == "-laplacian_loss_weight" ];then
        temp=9
        continue
    fi
    
    
    if [ 1 == $temp ];then
        #echo $temp
        array_lr[${#array_lr[*]}]=$i
    fi
    
    if [ 2 == $temp ];then
        #echo $temp
        array_epochs[${#array_epochs[*]}]=$i
    fi
    
    if [ 3 == $temp ];then
        #echo $temp
        array_hidden_dim[${#array_hidden_dim[*]}]=$i
    fi
    
    if [ 4 == $temp ];then
        #echo $temp
        array_side_hidden_dim[${#array_side_hidden_dim[*]}]=$i
    fi
    
    if [ 5 == $temp ];then
        #echo $temp
        array_dropout[${#array_dropout[*]}]=$i
    fi
    
    if [ 6 == $temp ];then
        #echo $temp
        array_use_side_feature[${#array_use_side_feature[*]}]=$i
    fi
    
    if [ 7 == $temp ];then
        #echo $temp
        array_use_data_whitening[${#array_use_data_whitening[*]}]=$i
    fi
    
    if [ 8 == $temp ];then
        #echo $temp
        array_use_laplacian_loss[${#array_use_laplacian_loss[*]}]=$i
    fi
    
    if [ 9 == $temp ];then
        #echo $temp
        array_laplacian_loss_weight[${#array_laplacian_loss_weight[*]}]=$i
    fi
    
done



for i in ${array_lr[@]}
do
    for j in ${array_epochs[@]}
    do
        for k in ${array_hidden_dim[@]}
        do
            for l in ${array_side_hidden_dim[@]}
            do
                for m in ${array_dropout[@]}
                do
                    for n in ${array_use_side_feature[@]}
                    do
                        for o in ${array_use_data_whitening[@]}
                        do
                            for p in ${array_use_laplacian_loss[@]}
                            do

                                for q in ${array_laplacian_loss_weight[@]}
                                do
                                echo "##############################################################################################################################################################"
                                echo "learning rate:$i epochs:$j hidden_dim:$k side_hidden_dim:$l dropout:$m use_side_feature:$n use_data_whitening:$o use_laplacian_loss:$p laplacian_loss_weight:$q"
                                python3 run.py --train_flag 1 \
                                                --test_flag 1 \
                                                --rate_num 5 \
                                                --lr $i \
                                                --weight_decay 0.00001 \
                                                --num_epochs $j \
                                                --hidden_dim $k \
                                                --side_hidden_dim $l \
                                                --out_dim 5 \
                                                --drop_out $m \
                                                --split_ratio 0.8 \
                                                --save_steps 100 \
                                                --log_dir './log' \
                                                --saved_model_folder './parameters' \
                                                --use_side_feature $n \
                                                --use_data_whitening $o \
                                                --use_laplacian_loss $p \
                                                --laplacian_loss_weight $q
                                echo "##############################################################################################################################################################"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done