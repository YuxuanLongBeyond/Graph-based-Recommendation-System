#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:26:33 2019

@author: YuxuanLong
"""

import numpy as np
import torch
import utils
from utils import epsilon_similarity_graph, compute_laplacian
import torch.optim as optim
import torch.nn as nn
import os
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import sys
import os 
import json
from dataset import prepare

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--rate_num', type=int, default=5, help='todo')
parser.add_argument('--use_side_feature', default=0, type=int, help='using side feature')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay rate')
parser.add_argument('--num_epochs', type=int, default=1000, help='number of training epochs')
parser.add_argument('--hidden_dim', type=int, default=5, help='hidden dimension')
parser.add_argument('--side_hidden_dim', type=int, default=5, help='side hidden dimension')
parser.add_argument('--out_dim', type=int, default=5, help='output dimension')
parser.add_argument('--drop_out', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--split_ratio', type=float, default=0.8, help='split ratio for training set')
parser.add_argument('--save_steps', type=int, default=100, help='every #steps to save the model')
parser.add_argument('--log_dir', help='folder to save log')
parser.add_argument('--saved_model_folder', help='folder to save model')
parser.add_argument('--use_data_whitening', default=0, type=int, help='data whitening')
parser.add_argument('--use_laplacian_loss', default=0, type=int, help='laplacian loss')
parser.add_argument('--laplacian_loss_weight', default=0.1, type=float, help='laplacian loss weight')
parser.add_argument('--dataset_path', help='dataset path')
parser.add_argument('--save_processed_data_path', help='path to save the processed data')

args = parser.parse_args()


# Run on GPU if CUDA is available
RUN_ON_GPU = torch.cuda.is_available()

# Set random seeds
SEED = 2019
np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)

def validate(score, rate_num, user_item_matrix_test):
    sm = nn.Softmax(dim = 0)
    score = sm(score)
    score_list = torch.split(score, rate_num)
    pred = 0
    for i in range(rate_num):
        pred += (i + 1) * score_list[0][i]

    pred = utils.var_to_np(pred)
    
#     pred = np.load('./prediction.npy')
    
    ### test the performance
#     user_item_matrix_test = np.load('./processed_dataset/user_item_matrix_test.npy')
    test_mask = user_item_matrix_test > 0

    square_err = (pred * test_mask - user_item_matrix_test) ** 2
    mse = square_err.sum() / test_mask.sum()
    test_rmse = np.sqrt(mse)
    
    return test_rmse



def main(args):
    
    # get arguments
    rate_num = args.rate_num
    use_side_feature = args.use_side_feature
    lr = args.lr
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    hidden_dim = args.hidden_dim
    side_hidden_dim = args.side_hidden_dim
    out_dim = args.out_dim
    drop_out = args.drop_out
    split_ratio = args.split_ratio
    save_steps = args.save_steps
    log_dir = args.log_dir
    saved_model_folder =  args.saved_model_folder
    use_data_whitening = args.use_data_whitening
    use_laplacian_loss = args.use_laplacian_loss
    laplacian_loss_weight = args.laplacian_loss_weight
    
    # mark and record the training file, save the training arguments for future analysis
    post_fix = '/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_dir = log_dir + post_fix
    writer = SummaryWriter(log_dir=log_dir)
    f = open(log_dir + '/test.txt', 'a')
    f.write(str(vars(args)))
    f.close()
    
    #get prepared data
    feature_u, feature_v, feature_dim, all_M_u, all_M_v, side_feature_u, side_feature_v, all_M, mask, user_item_matrix_train, user_item_matrix_test, laplacian_u, laplacian_v = prepare(args)  
    

    if not os.path.exists(saved_model_folder):
        os.makedirs(saved_model_folder)  
    weights_name = saved_model_folder + post_fix + '_weights'


    net = utils.create_models(feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                 side_hidden_dim, side_feature_u, side_feature_v, use_side_feature, out_dim, drop_out)
    net.train() # in train mode

    # create AMSGrad optimizer
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay)
    Loss = utils.loss(all_M, mask, user_item_matrix_train, laplacian_loss_weight)
    iter_bar = tqdm(range(num_epochs), desc='Iter (loss=X.XXX)')
    for epoch in iter_bar:

        optimizer.zero_grad()

        score = net.forward()

        if use_laplacian_loss:
            loss = Loss.laplacian_loss(score, laplacian_u, laplacian_v)
        else:
             loss = Loss.loss(score)

        loss.backward()

        optimizer.step()

        with torch.no_grad():
            rmse = Loss.rmse(score)
            
            val_rmse = validate(score, rate_num, user_item_matrix_test)
            iter_bar.set_description('Iter (loss=%5.3f, rmse=%5.3f, val_rmse=%5.3f)'%(loss.item(),rmse.item(), val_rmse.item()))

        writer.add_scalars('data/scalar_group',{'loss': loss.item(), 'rmse': rmse.item(), 'val_rmse':val_rmse.item()},epoch)

        if epoch % save_steps == 0:
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
    
#     pred = np.load('./prediction.npy')
    
    ### test the performance
#     user_item_matrix_test = np.load('./processed_dataset/user_item_matrix_test.npy')
    test_mask = user_item_matrix_test > 0

    square_err = (pred * test_mask - user_item_matrix_test) ** 2
    mse = square_err.sum() / test_mask.sum()
    test_rmse = np.sqrt(mse)
    print('Test RMSE: ', test_rmse)

    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)