# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:36:26 2021

@author: groes
"""
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from abc import ABC, abstractmethod

class PiNet(nn.Module):
    """
        This class is the actor network. It evaluates the value of a state and 
        action pair. 
        
        lr : learning rate
        input_dims : numbre of input dimensions from env
        
        activation_func : STR
            The activation function to be used. Should be passed as string, e.g.
            "T.relu(input)". It will be evaluated at runtime via eval()
    """
    
    def __init__(self, lr, input_dims, num_actions, name, activation_func, 
                 checkpoint_dir = "tmp/sac", fc1_dims=256, fc2_dims=256):
        super().__init__()
        
        self.input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.num_actions = num_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_sac.pt")
        self.activation_function = activation_func
        
        self.fc1 = nn.Linear(self.input_dims[0], self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, self.num_actions)
    


    
    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = self.activation(self.activation_func)
        action_value = self.fc2(action_value)
        action_value = self.activation(self.activation_func)
        q = self.q(action_value)
        
        return q
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))