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
                 all_M, side_hidden_dim, side_feature_u, side_feature_v, out_dim):
        super(GCMC, self).__init__()
        ###To Do:
        #### drop out layer
        #### M normalization
        #### regularization on Q
        
        ## add batch normalization?

        self.feature_u = feature_u
        self.feature_v = feature_v
        self.rate_num = rate_num
        
        self.side_feature_u = side_feature_u
        self.side_feature_v = side_feature_v
        
        self.W = nn.Parameter(torch.randn(rate_num, feature_dim, hidden_dim))
        
        self.all_M = all_M
        self.reLU = nn.ReLU()
        
        self.linear_layer_side_u = nn.Linear(side_feature_dim, side_hidden_dim, bias = True)
        self.linear_layer_side_v = nn.Linear(side_feature_dim, side_hidden_dim, bias = True)
        
        self.linear_cat_u = nn.Linear(hidden_dim + side_hidden_dim, out_dim, bias = False)
        self.linear_cat_v = nn.Linear(hidden_dim + side_hidden_dim, out_dim, bias = False)
        
        self.Q = nn.Parameter(torch.randn(rate_num, out_dim, out_dim))
        
    def forward(self):
        hidden_feature_u = []
        hidden_feature_v = []
        
        W_list = torch.split(self.W, self.rate_num)
        M_list = torch.split(self.all_M, self.rate_num)
        
        for i in range(self.rate_num):
            Wr = W_list[0][i]
            M = M_list[0][i]
            hidden_u = sp.mm(self.feature_v, Wr)
            hidden_u = self.reLU(sp.mm(M, hidden_u))
            
            ### need to further process M, normalization
            hidden_v = sp.mm(self.feature_u, Wr)
            hidden_v = self.reLU(sp.mm(torch.t(M), hidden_v))
            
            hidden_feature_u.append(hidden_u)
            hidden_feature_v.append(hidden_v)
        hidden_feature_u = torch.cat(hidden_feature_u, 0)
        hidden_feature_v = torch.cat(hidden_feature_v, 0)
        
        
        side_hidden_feature_u = self.reLU(self.linear_layer_side_u(self.side_feature_u))
        side_hidden_feature_v = self.reLU(self.linear_layer_side_v(self.side_feature_v))
        
        cat_u = torch.cat((hidden_feature_u, side_hidden_feature_u), 1)
        cat_v = torch.cat((hidden_feature_v, side_hidden_feature_v), 1)
        
        embed_u = self.reLU(self.linear_cat_u(cat_u))
        embed_v = self.reLU(self.linear_cat_v(cat_v))
        
        score = []
        Q_list = torch.split(self.Q, self.rate_num)
        for i in range(self.rate_num):
            Qr = Q_list[0][i]
            
            tem = torch.exp(torch.mm(torch.mm(embed_u, Qr), torch.t(embed_v)))
            
            score.append(tem)
            
        score = torch.stack(score), dim = 0
            
class Loss(nn.Module):
    def __init__(self, all_M):
            
        super(Loss, self).__init__()
            
        self.target = torch.argmax(all_M, 0)
        self.CE = torch.nn.CrossEntropyLoss()
            
    def loss(self, score):
        return self.CE(score, self.target)
            
            
            
            
            
            
            
            
            
            
            
