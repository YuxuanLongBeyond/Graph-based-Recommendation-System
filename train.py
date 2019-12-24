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
    
    # rating 1
    user_item_matrix_rating_1 = user_item_matrix==1
    # rating 2
    user_item_matrix_rating_2 = user_item_matrix==2
    # rating 3
    user_item_matrix_rating_3 = user_item_matrix==3
    # rating 4
    user_item_matrix_rating_4 = user_item_matrix==4
    # rating 5
    user_item_matrix_rating_5 = user_item_matrix==5
    
    num_user, num_item = user_item_matrix.shape
    
    side_feature_u = np.random.randn(num_user, 10)
    side_feature_v = np.random.randn(num_item, 20)
    
    rate_num = 5
    I = np.eye(num_user + num_item)
    feature_u = I[0:num_user, :]
    feature_v = I[0:num_item, :]
    feature_dim = num_user + num_item

    
    all_M_u = []
    all_M_v = []
    all_M = []
    for i in range(5):
        M_r = user_item_matrix==i
        all_M_u.append(utils.normalize(M_r))
        all_M_v.append(utils.normalize(M_r.T))
        
        all_M.append(M_r)
    
    all_M = np.array(all_M)
    mask = user_item_matrix > 0

    if not os.path.exists('./parameters'):
        os.makedirs('./parameters')  
    weights_name = './parameters/weights'
    
    lr = 1e-4
    weight_decay = 1e-5
    num_epochs = 10
    hidden_dim = 10
    side_hidden_dim = 10
    out_dim = 10
    
    net = utils.create_models(feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                 side_hidden_dim, side_feature_u, side_feature_v, out_dim)
    net.train() # in train mode
    
    
    
    # create AMSGrad optimizer
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay, amsgrad = True)
    Loss = utils.loss(all_M, mask)

    for epoch in range(num_epochs):
        print('Start epoch ', epoch)
        
        optimizer.zero_grad()

        pred = net.forward()

        loss = Loss.loss(pred)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss = loss.data.item()
        print('Loss: ', epoch_loss)

        
    torch.save(net.state_dict(), weights_name)
