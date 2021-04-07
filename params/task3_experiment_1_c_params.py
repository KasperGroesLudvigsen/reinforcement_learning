# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:00:59 2021

@author: hugha
"""

import torch.nn as nn


environment_params = {
    'N' : 10,
    'stumble_prob' : 0.2,
    'observation_size' : 10
    }

buffer_params = {
    'obs_dims': (1,125)
    }

ac_params = {
    'num_obs' : 125,
    'num_actions' : 10,
    'hidden_sizes' : [256,256],
    'activation_func': nn.ReLU
     }

learning_params = {
    'lr': 0.0003,
    'gamma': 0.9,
    'batch_size': 64,
    'polyak' : 0.05,
    'clipping_norm': 1,
    "automatic_entropy_tuning":False,
    "entropy_alpha":0.3,
    "number_of_episodes" : 10000, 
    "ere" : True
    }

PARAMS = {
    "experiment_name" : "experiment_name",
    "environment_params" : environment_params, 
    "buffer_params" : buffer_params,
    "ac_params" : ac_params,
    "learning_params" : learning_params
    }