#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:58:25 2019

@author: YuxuanLong
"""

import numpy as np
import torch
from torch.autograd import Variable
import model


# Run on GPU if CUDA is available
RUN_ON_GPU = torch.cuda.is_available()

# Set random seeds
SEED = 2019
np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)


def np_to_var(x):
    '''
    Convert numpy array to Torch variable.
    '''
    if RUN_ON_GPU:
        x = x.cuda()
    return Variable(x)


def var_to_np(x):
    '''
    Convert Torch variable to numpy array.
    '''
    if RUN_ON_GPU:
        x = x.cpu()
    return x.data.numpy()

def normalize(M):
    return (M.T / np.sum(M, axis = 1)).T


def create_models(feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                 side_hidden_dim, side_feature_u, side_feature_v, out_dim, drop_out = 0.0):
    '''
    Choose one model from our implementations
    '''
    side_feature_u = np_to_var(side_feature_u)
    side_feature_v = np_to_var(side_feature_v)
    
    all_M_u = np_to_var(all_M_u).to_sparse()
    all_M_v = np_to_var(all_M_v).to_sparse()
    
    feature_u = np_to_var(feature_u).to_sparse()
    feature_v = np_to_var(feature_v).to_sparse()
    
    
    net = model.GCMC(feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                 side_hidden_dim, side_feature_u, side_feature_v, out_dim)

    if RUN_ON_GPU:
        print('Moving models to GPU.')
        net.cuda()
    else:
        print('Keeping models on CPU.')

    return net



def loss(all_M):
    return model.Loss(all_M)