#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:57:42 2019

@author: YuxuanLong
"""


import numpy as np
import torch
import os
import utils

# Run on GPU if CUDA is available
RUN_ON_GPU = torch.cuda.is_available()

# Set random seeds
SEED = 2019
np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)


if __name__ == '__main__':
    user_item_matrix_test = np.load('./processed_dataset/user_item_matrix_test.npy')
    
    
    lr = 1e-3 # 1e-2
    weight_decay = 1e-5
    num_epochs = 2000 # 1000
    hidden_dim = 50 # 100
    side_hidden_dim = 20 # 10
    out_dim = 75 # 75
    
    I = np.eye(num_user + num_item)
    feature_u = I[0:num_user, :]
    feature_v = I[num_user:, :]
    
    net = utils.create_models(feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                 side_hidden_dim, side_feature_u, side_feature_v, use_side, out_dim)
    
    net = utils.create_models(model_choice)
    linkNet = None
    DlinkNet = None
    
    weights_name = './parameters/weights' + str(model_choice)
#    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    if RUN_ON_GPU:
        net.load_state_dict(torch.load(weights_name))
    else:
        net.load_state_dict(torch.load(weights_name, map_location = lambda storage, loc: storage))
    net.eval()