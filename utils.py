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
    """
    Convert numpy array to Torch variable.
    """
    x = torch.from_numpy(x)
    if RUN_ON_GPU:
        x = x.cuda()
    return Variable(x)


def var_to_np(x):
    """
    Convert Torch variable to numpy array.
    """
    if RUN_ON_GPU:
        x = x.cpu()
    return x.data.numpy()

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def normalize(M):
    s = np.sum(M, axis = 1)
    s[s == 0] = 1
    return (M.T / s).T


def create_models(feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                 side_hidden_dim, side_feature_u, side_feature_v, use_side, out_dim, drop_out = 0.0):
    """
    Choose one model from our implementations
    """
    side_feature_u = np_to_var(side_feature_u.astype(np.float32))
    side_feature_v = np_to_var(side_feature_v.astype(np.float32))
    
    for i in range(rate_num):
        all_M_u[i] = to_sparse(np_to_var(all_M_u[i].astype(np.float32)))
        all_M_v[i] = to_sparse(np_to_var(all_M_v[i].astype(np.float32)))
#        all_M_u[i] = np_to_var(all_M_u[i].astype(np.float32))
#        all_M_v[i] = np_to_var(all_M_v[i].astype(np.float32))   
    
    feature_u = to_sparse(np_to_var(feature_u.astype(np.float32)))
    feature_v = to_sparse(np_to_var(feature_v.astype(np.float32)))
#    feature_u = np_to_var(feature_u.astype(np.float32))
#    feature_v = np_to_var(feature_v.astype(np.float32))

    net = model.GCMC(feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                 side_hidden_dim, side_feature_u, side_feature_v, use_side, out_dim)

    if RUN_ON_GPU:
        print('Moving models to GPU.')
        net.cuda()
    else:
        print('Keeping models on CPU.')

    return net



def loss(all_M, mask, user_item_matrix):
    all_M = np_to_var(all_M.astype(np.float32))
    mask = np_to_var(mask.astype(np.float32))
    user_item_matrix = np_to_var(user_item_matrix.astype(np.float32))
    
    return model.Loss(all_M, mask, user_item_matrix)