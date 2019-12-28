#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:26:33 2019

@author: YuxuanLong
"""

import numpy as np
import torch
import utils
import torch.optim as optim
import torch.nn as nn
import os

# Run on GPU if CUDA is available
RUN_ON_GPU = torch.cuda.is_available()

# Set random seeds
SEED = 2019
np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)


if __name__ == '__main__':
    user_item_matrix = np.load('./processed_dataset/user_item_matrix.npy')
    num_user, num_item = user_item_matrix.shape
    
    split_ratio = 0.8
    
    mask = user_item_matrix > 0
    mask_new = mask + np.random.uniform(0, 1, (num_user, num_item))
    train_mask = (mask_new <= (1 + split_ratio)) & mask
    test_mask = (mask_new > (1 + split_ratio)) & mask
    
    user_item_matrix_train = user_item_matrix + 0
    user_item_matrix_train[test_mask] = 0
    
    user_item_matrix_test = user_item_matrix + 0
    user_item_matrix_test[train_mask] = 0
    
    np.save('./processed_dataset/user_item_matrix_train.npy', user_item_matrix_train)
    np.save('./processed_dataset/user_item_matrix_test.npy', user_item_matrix_test)
    
    side_feature_u = np.random.randn(num_user, 10)
    side_feature_v = np.random.randn(num_item, 20)
    
    rate_num = 5

    feature_dim = num_user + num_item

    I = np.eye(num_user + num_item)
    feature_u = I[0:num_user, :]
    feature_v = I[num_user:, :]
    
    all_M_u = []
    all_M_v = []
    all_M = []
    for i in range(rate_num):
        M_r = user_item_matrix_train == (i + 1)
        all_M_u.append(utils.normalize(M_r))
        all_M_v.append(utils.normalize(M_r.T))
        
        all_M.append(M_r)
    
    all_M = np.array(all_M)
    mask = user_item_matrix_train > 0

    if not os.path.exists('./parameters'):
        os.makedirs('./parameters')  
    weights_name = './parameters/weights'
    
    save_period = 1
    use_side = False
    
    lr = 1e-3 # 1e-2
    weight_decay = 1e-5
    num_epochs = 1000 # 1000
    hidden_dim = 5 # 100
    side_hidden_dim = 5 # 10
    out_dim = 5 # 75
    
    net = utils.create_models(feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                 side_hidden_dim, side_feature_u, side_feature_v, use_side, out_dim)
    net.train() # in train mode

    # create AMSGrad optimizer
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay, amsgrad = True)
    Loss = utils.loss(all_M, mask, user_item_matrix_train)

    for epoch in range(num_epochs):
        
        optimizer.zero_grad()

        score = net.forward()

        loss = Loss.loss(score)
        
        loss.backward()
        
        optimizer.step()
#        print('Loss: ', loss.data.item())
        
        
        if epoch % 100 == 0:
            print('Start epoch ', epoch)
            epoch_loss = loss.data.item()
            print('Loss: ', epoch_loss)
            
        if epoch % save_period == 0:
            with torch.no_grad():
                rmse = Loss.rmse(score)
                print('Training RMSE: ', rmse.data.item())        
                                
                torch.save(net.state_dict(), weights_name)

    rmse = Loss.rmse(score)
    print('Final training RMSE: ', rmse.data.item())        
    torch.save(net.state_dict(), weights_name)
    
    
    sm = nn.Softmax(dim = 0)
    score = sm(score)
    score_list = torch.split(score, rate_num)
    pred = 0
    for i in range(rate_num):
        pred += (i + 1) * score_list[0][i]

    pred = utils.var_to_np(pred)
    np.save('./prediction.npy', pred)
    
    ### test
    
    test_mask = user_item_matrix_test > 0
    
    square_err = (pred * test_mask - user_item_matrix_test) ** 2
    mse = square_err.sum() / test_mask.sum()
    test_rmse = np.sqrt(mse)
    print('Test RMSE: ', test_rmse)

