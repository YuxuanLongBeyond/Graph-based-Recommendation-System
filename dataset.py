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


### input feature generation
feature_dim = num_user + num_item
I = np.eye(num_user + num_item)
feature_u = I[0:num_user, :]
feature_v = I[num_user:, :]