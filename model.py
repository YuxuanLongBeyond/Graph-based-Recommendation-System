#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:02:15 2019

@author: YuxuanLong
"""

import torch
import torch.nn as nn
import torch.sparse as sp


class GCMC(nn.Module):
    def __init__(self, feature_u, feature_v, feature_dim, hidden_dim, side_feature_dim, rate_num, 
                 M_list, side_hidden_dim, side_feature_u, side_feature_v, out_dim):
        super(GCMC, self).__init__()
        ###To Do:
        #### drop out layer
        #### M normalization
        #### bilinear decoder
        
        ## add batch normalization?

        self.feature_u = feature_u
        self.feature_v = feature_v
        self.rate_num = rate_num
        
        self.side_feature_u = side_feature_u
        self.side_feature_v = side_feature_v
        
        self.W = nn.Parameter(torch.randn(rate_num, feature_dim, hidden_dim))
        
        self.M_list = M_list
        self.linear_layer
        self.reLU = nn.ReLU()
        
        self.linear_layer_side = nn.Linear(side_feature_dim, side_hidden_dim, bias = False)
        self.bias_u = nn.Parameter(torch.randn(side_hidden_dim))
        self.bias_v = nn.Parameter(torch.randn(side_hidden_dim))
        
        self.linear_cat_u = nn.Linear(hidden_dim + side_hidden_dim, out_dim, bias = False)
        self.linear_cat_v = nn.Linear(hidden_dim + side_hidden_dim, out_dim, bias = False)
        
    def forward(self):
        hidden_feature_u = []
        hidden_feature_v = []
        
        W_list = torch.split(self.W)
        for i in range(self.rate_num):
            Wr = W_list[i]
            M = self.M_list[i]
            hidden_u = sp.mm(self.feature_v, Wr)
            hidden_u = self.reLU(sp.mm(M, hidden_u))
            
            ### need to further process M, normalization
            hidden_v = sp.mm(self.feature_u, Wr)
            hidden_v = self.reLU(sp.mm(M.T, hidden_v))
            
            hidden_feature_u.append(hidden_u)
            hidden_feature_v.append(hidden_v)
        hidden_feature_u = torch.cat(hidden_feature_u, 0)
        hidden_feature_v = torch.cat(hidden_feature_v, 0)
        
        
        side_hidden_feature_u = self.reLU(self.linear_layer_side(self.side_feature_u) + self.bias_u)
        side_hidden_feature_v = self.reLU(self.linear_layer_side(self.side_feature_v) + self.bias_v)
        
        cat_u = torch.cat((hidden_feature_u, side_hidden_feature_u), 1)
        cat_v = torch.cat((hidden_feature_v, side_hidden_feature_v), 1)
        
        embed_u = self.reLU(self.linear_cat_u(cat_u))
        embed_v = self.reLU(self.linear_cat_v(cat_v))
            
