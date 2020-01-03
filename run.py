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
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import sys
import os 
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_flag', default=0, type=int, help='training flag')
parser.add_argument('--test_flag', default=0, type=int, help='test flag')
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

args = parser.parse_args()



# Run on GPU if CUDA is available
RUN_ON_GPU = torch.cuda.is_available()

# Set random seeds
SEED = 2019
np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)

def data_whitening(x, epsilon = 1e-9):
    """
    Perform ZCA for data whitening on the features.
    
    Return:
        M       the linear transformation matrix
        mean    the mean of the feature
    """
    mean = np.mean(x, axis = 0)
    x_norm = x - mean
    sigma = np.dot(x_norm.T, x_norm) / x.shape[0]
    u, V = np.linalg.eig(sigma)
    M = np.dot(V / np.sqrt(u + epsilon), V.T)
    return M, mean


def epsilon_similarity_graph(X: np.ndarray, sigma=None, epsilon=0):
    """ X (n x d): coordinates of the n data points in R^d.
        sigma (float): width of the kernel
        epsilon (float): threshold
        Return:
        adjacency (n x n ndarray): adjacency matrix of the graph.
    """
    # Your code here
    W = np.array([np.sum((X[i] - X)**2, axis = 1) for i in range(X.shape[0])])
    typical_dist = np.mean(np.sqrt(W))
    # print(np.mean(W))
    c = 0.35
    if sigma == None:
        sigma = typical_dist * c
    
    mask = W >= epsilon
    
    adjacency = np.exp(- W / 2.0 / (sigma ** 2))
    adjacency[mask] = 0.0
    adjacency -= np.diag(np.diag(adjacency))
    return adjacency

def compute_laplacian(adjacency: np.ndarray, normalize: bool):
    """ Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    # Your code here
    d = np.sum(adjacency, axis = 1)
    d_sqrt = np.sqrt(d)  
#     print("d_sqrt {}".format(d_sqrt))
    D = np.diag(1 / d_sqrt)
    if normalize:
        L = np.eye(adjacency.shape[0]) - (adjacency.T / d_sqrt).T / d_sqrt
        # L = np.dot(np.dot(D, np.diag(d) - adjacency), D)
    else:
        L = np.diag(d) - adjacency
    return L


def main(args):
    
    train_flag = args.train_flag
    test_flag = args.test_flag
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
    
    post_fix = '/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_dir = log_dir + post_fix
    writer = SummaryWriter(log_dir=log_dir)
    f = open(log_dir + '/test.txt', 'a')
    f.write(str(vars(args)))
    f.close()
    
    ### Rating matrix loading, processing, split
    user_item_matrix = np.load('./processed_dataset/user_item_matrix.npy')
    num_user, num_item = user_item_matrix.shape
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
    
    
    ### side feature loading and processing
    raw_side_feature_u = np.load('./processed_dataset/user_data_np.npy', allow_pickle = True)
    raw_side_feature_v = np.load('./processed_dataset/item_data_np.npy', allow_pickle = True)
    
    if use_data_whitening:
        print("Using data whitening!")
    
        M_u, mean_u = data_whitening(raw_side_feature_u)
        M_v, mean_v = data_whitening(raw_side_feature_v)
        side_feature_u = np.dot(raw_side_feature_u - mean_u, M_u)
        side_feature_v = np.dot(raw_side_feature_v - mean_v, M_v)
        
    else:
        print("Not using data whitening!")
        side_feature_u = raw_side_feature_u
        side_feature_v = raw_side_feature_v
    
    ############test############
    
#     np.save("side_feature_u_whitening.npy", side_feature_u)
#     np.save("side_feature_v_whitening.npy", side_feature_v)
    
    if use_data_whitening:
        adjacency_u = epsilon_similarity_graph(side_feature_u, epsilon=5.7)
        laplacian_u = compute_laplacian(adjacency_u, True)
        adjacency_v = epsilon_similarity_graph(side_feature_v, epsilon=52.5)
        laplacian_v = compute_laplacian(adjacency_v, True)
        
    else:
        adjacency_u = epsilon_similarity_graph(side_feature_u, epsilon=1.1)
        laplacian_u = compute_laplacian(adjacency_u, True)
        adjacency_v = epsilon_similarity_graph(side_feature_v, epsilon=2.1)
        laplacian_v = compute_laplacian(adjacency_v, True)
    
    laplacian_u = utils.np_to_var(laplacian_u)
    laplacian_v = utils.np_to_var(laplacian_v)
#     print(type(laplacian_u))
    
    
    ### input feature generation
    feature_dim = num_user + num_item
    I = np.eye(num_user + num_item)
    feature_u = I[0:num_user, :]
    feature_v = I[num_user:, :]
    
#     print("test here")
#     print(train_flag == True)
    
    if train_flag:
        if not os.path.exists(saved_model_folder):
            os.makedirs(saved_model_folder)  
        weights_name = saved_model_folder + post_fix + '_weights'
        
    
        net = utils.create_models(feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                     side_hidden_dim, side_feature_u, side_feature_v, use_side_feature, out_dim, drop_out)
        net.train() # in train mode
    
        # create AMSGrad optimizer
        optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay)
        Loss = utils.loss(all_M, mask, user_item_matrix_train, laplacian_loss_weight)
#         iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
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
                iter_bar.set_description('Iter (loss=%5.3f, rmse=%5.3f)'%(loss.item(),rmse.item()))
                
            writer.add_scalars('data/scalar_group',{'loss': loss.item(), 'rmse': rmse.item()},epoch)
                                                    
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
        np.save('./prediction.npy', pred)
    
    ### test
    if test_flag:
        pred = np.load('./prediction.npy')
        
        test_mask = user_item_matrix_test > 0
        
        square_err = (pred * test_mask - user_item_matrix_test) ** 2
        mse = square_err.sum() / test_mask.sum()
        test_rmse = np.sqrt(mse)
        print('Test RMSE: ', test_rmse)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)