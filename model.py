#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:02:15 2019

@author: YuxuanLong
"""

import torch
import torch.nn as nn
import torch.sparse as sp


def sparse_drop(feature, drop_out):
    tem = torch.rand((feature._nnz()))
    feature._values()[tem < drop_out] = 0
    return feature

class GCMC(nn.Module):
    def __init__(self, feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                 side_hidden_dim, side_feature_u, side_feature_v, use_side, out_dim, drop_out = 0.0):
        super(GCMC, self).__init__()
        ###To Do:
        #### regularization on Q
        
        self.drop_out = drop_out
        
        side_feature_u_dim = side_feature_u.shape[1]
        side_feature_v_dim = side_feature_v.shape[1]
        self.use_side = use_side

        self.feature_u = feature_u
        self.feature_v = feature_v
        self.rate_num = rate_num
        
        self.num_user = feature_u.shape[0]
        self.num_item = feature_v.shape[1]
        
        self.side_feature_u = side_feature_u
        self.side_feature_v = side_feature_v
        
        self.W = nn.Parameter(torch.randn(rate_num, feature_dim, hidden_dim))
        nn.init.kaiming_normal_(self.W, mode = 'fan_out', nonlinearity = 'relu')
        
        self.all_M_u = all_M_u
        self.all_M_v = all_M_v
        
        self.reLU = nn.ReLU()
        
        if use_side:
            self.linear_layer_side_u = nn.Sequential(*[nn.Linear(side_feature_u_dim, side_hidden_dim, bias = True), 
                                                       nn.BatchNorm1d(side_hidden_dim), nn.ReLU()])
            self.linear_layer_side_v = nn.Sequential(*[nn.Linear(side_feature_v_dim, side_hidden_dim, bias = True), 
                                                       nn.BatchNorm1d(side_hidden_dim), nn.ReLU()])
    
            self.linear_cat_u = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2 + side_hidden_dim, out_dim, bias = True), 
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])
            self.linear_cat_v = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2 + side_hidden_dim, out_dim, bias = True), 
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])    
        else:
            
            self.linear_cat_u = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2, out_dim, bias = True), 
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])
            self.linear_cat_v = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2, out_dim, bias = True), 
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])

#        self.linear_cat_u2 = nn.Sequential(*[nn.Linear(out_dim, out_dim, bias = True), 
#                                            nn.BatchNorm1d(out_dim), nn.ReLU()])
#        self.linear_cat_v2 = nn.Sequential(*[nn.Linear(out_dim, out_dim, bias = True), 
#                                            nn.BatchNorm1d(out_dim), nn.ReLU()])
    
        self.Q = nn.Parameter(torch.randn(rate_num, out_dim, out_dim))
        nn.init.orthogonal_(self.Q)
        
        
    def forward(self):
        
        feature_u_drop = sparse_drop(self.feature_u, self.drop_out) / (1.0 - self.drop_out)
        feature_v_drop = sparse_drop(self.feature_v, self.drop_out) / (1.0 - self.drop_out)
        
        hidden_feature_u = []
        hidden_feature_v = []
        
        W_list = torch.split(self.W, self.rate_num)
        W_flat = []
        for i in range(self.rate_num):
            Wr = W_list[0][i]
            M_u = self.all_M_u[i]
            M_v = self.all_M_v[i]
            hidden_u = sp.mm(feature_v_drop, Wr)
            hidden_u = self.reLU(sp.mm(M_u, hidden_u))
            
            ### need to further process M, normalization
            hidden_v = sp.mm(feature_u_drop, Wr)
            hidden_v = self.reLU(sp.mm(M_v, hidden_v))

            
            hidden_feature_u.append(hidden_u)
            hidden_feature_v.append(hidden_v)
            
            W_flat.append(Wr)
            
        hidden_feature_u = torch.cat(hidden_feature_u, dim = 1)
        hidden_feature_v = torch.cat(hidden_feature_v, dim = 1)
        W_flat = torch.cat(W_flat, dim = 1)


        cat_u = torch.cat((hidden_feature_u, torch.mm(self.feature_u, W_flat)), dim = 1)
        cat_v = torch.cat((hidden_feature_v, torch.mm(self.feature_v, W_flat)), dim = 1)
        
        if self.use_side:
            side_hidden_feature_u = self.linear_layer_side_u(self.side_feature_u)
            side_hidden_feature_v = self.linear_layer_side_v(self.side_feature_v)    
            
            cat_u = torch.cat((cat_u, side_hidden_feature_u), dim = 1)
            cat_v = torch.cat((cat_v, side_hidden_feature_v), dim = 1)
        
        
        embed_u = self.linear_cat_u(cat_u)
        embed_v = self.linear_cat_v(cat_v)
        
        score = []
        Q_list = torch.split(self.Q, self.rate_num)
        for i in range(self.rate_num):
            Qr = Q_list[0][i]
            
            tem = torch.mm(torch.mm(embed_u, Qr), torch.t(embed_v))
            
            score.append(tem)
        return torch.stack(score)
            
class Loss(nn.Module):
    def __init__(self, all_M, mask, user_item_matrix):
            
        super(Loss, self).__init__()
            
        self.all_M = all_M
        self.mask = mask
        self.user_item_matrix = user_item_matrix
        
        self.rate_num = all_M.shape[0]
        self.num = float(mask.sum())
        
        self.logsm = nn.LogSoftmax(dim = 0)
        self.sm = nn.Softmax(dim = 0)
        
    def cross_entropy(self, score):
        l = torch.sum(-self.all_M * self.logsm(score))
        return l / self.num
    
    def rmse(self, score):
        score_list = torch.split(self.sm(score), self.rate_num)
        total_score = 0
        for i in range(self.rate_num):
            total_score += (i + 1) * score_list[0][i]
        
        square_err = torch.pow(total_score * self.mask - self.user_item_matrix, 2)
        mse = torch.sum(square_err) / self.num
        return torch.sqrt(mse)
        
    def loss(self, score):
        return self.cross_entropy(score) + self.rmse(score)

        


