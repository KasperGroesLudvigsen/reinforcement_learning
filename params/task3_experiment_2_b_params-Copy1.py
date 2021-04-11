# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 18:07:55 2021

@author: hugha
"""

import torch.nn as nn


environment_params = {
    'N' : 8,
    'stumble_prob' : 0.5,
    'observation_size' : 0
    }

buffer_params = {
    'obs_dims': (1,5)
    }

ac_params = {
    'num_obs' : 5,
    'num_actions' : 10,
    'hidden_sizes' : [256,256],
    'activation_func': nn.ReLU
     }

learning_params = {
    'lr': 0.0003,
    'gamma': 0.3,
    'batch_size': 64,
    'polyak' : 0.05,
    'clipping_norm': 1,
    "automatic_entropy_tuning":False,
    "entropy_alpha":0.3,
    "number_of_episodes" : 5000, 
    "ere" : False
    }

PARAMS = {
    "experiment_name" : "experiment_name",
    "environment_params" : environment_params, 
    "buffer_params" : buffer_params,
    "ac_params" : ac_params,
    "learning_params" : learning_params
    }