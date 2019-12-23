#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:57:42 2019

@author: YuxuanLong
"""


import numpy as np
import torch
import os



# Run on GPU if CUDA is available
RUN_ON_GPU = torch.cuda.is_available()
