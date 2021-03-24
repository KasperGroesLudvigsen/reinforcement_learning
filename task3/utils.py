# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:46:25 2021

@author: groes
"""

import torch.nn as nn
import numpy as np
import torch

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def convert_state(observation, env_size = 10, norm = 2., device = "cpu"):
    # Adapted from lab 7 feedback
    """
    observation (dict) : dictionary with relative coordinates and agent's surroundings
    
    env_size (int) : length of the environment's sides
    
    norm (float) : some float, e.g. 4.0, by which to normalize surroundings
    
    device (torch object) : GPU or CPU
    
    """
    rel_coor_car = observation['relative_coordinates_car'].flatten()/env_size
    rel_coor_britney = observation['relative_coordinates_britney'].flatten()/env_size
    o = observation['surroundings'].flatten()/norm
    state_tensor = np.concatenate([rel_coor_car, rel_coor_britney, o])
    #state_tensor = torch.tensor(state_tensor).unsqueeze(0)
    state_tensor = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0)
    
    return state_tensor
    

