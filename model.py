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
    def __init__(self, feature_u, feature_v, feature_dim, hidden_dim, side_feature_u_dim, side_feature_v_dim, rate_num, 
                 all_M_u, all_M_v, side_hidden_dim, side_feature_u, side_feature_v, out_dim, drop_out):
        super(GCMC, self).__init__()
        ###To Do:
        #### drop out on sparse features
        #### regularization on Q
        #### add batch normalization?

        self.feature_u = feature_u
        self.feature_v = feature_v
        self.rate_num = rate_num
        
        self.side_feature_u = side_feature_u
        self.side_feature_v = side_feature_v
        
        self.W = nn.Parameter(torch.randn(rate_num, feature_dim, hidden_dim))
        
        self.all_M_u = all_M_u
        self.all_M_v = all_M_v
        
        self.reLU = nn.ReLU()
        
        self.linear_layer_side_u = nn.Linear(side_feature_u_dim, side_hidden_dim, bias = True)
        self.linear_layer_side_v = nn.Linear(side_feature_v_dim, side_hidden_dim, bias = True)
        
        self.linear_cat_u = nn.Linear(hidden_dim + side_hidden_dim, out_dim, bias = False)
        self.linear_cat_v = nn.Linear(hidden_dim + side_hidden_dim, out_dim, bias = False)
        
        self.Q = nn.Parameter(torch.randn(rate_num, out_dim, out_dim))
        
    def forward(self):
        hidden_feature_u = []
        hidden_feature_v = []
        
        W_list = torch.split(self.W, self.rate_num)
        M_u_list = torch.split(self.all_M_u, self.rate_num)
        M_v_list = torch.split(self.all_M_v, self.rate_num)
        
        for i in range(self.rate_num):
            Wr = W_list[0][i]
            M_u = M_u_list[0][i]
            M_v = M_v_list[0][i]
            hidden_u = sp.mm(self.feature_v, Wr)
            hidden_u = self.reLU(sp.mm(M_u, hidden_u))
            
            ### need to further process M, normalization
            hidden_v = sp.mm(self.feature_u, Wr)
            hidden_v = self.reLU(sp.mm(M_v, hidden_v))
            
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
            
            tem = torch.mm(torch.mm(embed_u, Qr), torch.t(embed_v))
            
            score.append(tem)
            
        return torch.stack(score)
            
class Loss(nn.Module):
    def __init__(self, M_rating):
            
        super(Loss, self).__init__()
            
        self.M_rating = M_rating
        self.CE = torch.nn.CrossEntropyLoss()
            
    def loss(self, score):
        return self.CE(score.unsqueeze(0), self.M_rating.unsqueeze(0))
            
            
            
            
            
            
            
            
            
            
            
