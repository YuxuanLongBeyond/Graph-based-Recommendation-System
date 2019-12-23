#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:26:33 2019

@author: YuxuanLong
"""

import numpy as np

if __name__ == '__main__':
    user_item_matrix = np.load('./processed_dataset/user_item_matrix.npy')
    
    # rating 1
    user_item_matrix_rating_1 = user_item_matrix[user_item_matrix==1]
    # rating 2
    user_item_matrix_rating_1 = user_item_matrix[user_item_matrix==2]
    # rating 3
    user_item_matrix_rating_1 = user_item_matrix[user_item_matrix==3]
    # rating 4
    user_item_matrix_rating_1 = user_item_matrix[user_item_matrix==4]
    # rating 5
    user_item_matrix_rating_1 = user_item_matrix[user_item_matrix==5]    