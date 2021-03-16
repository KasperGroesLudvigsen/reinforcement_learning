# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:36:26 2021

@author: groes
"""
import os
import torch
import torch.nn as nn
import utils



class QNet(nn.Module):
    # adapted from:
    # https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/sac/core.py#L91

    def __init__(self, num_obs, num_actions, hidden_sizes, activation_func, 
                 name, checkpoint_dir = "tmp/sac"):
        """
        This class is the critic network. It evaluates the value of a state and 
        action pair. 
        
        num_obs : number (int) of observations  from env
        
        num_actions : number (int) of actions available to the agent
        
        hidden_sizes : tuple of ints where each int represents the number of
            neurons in a hidden layer
        
        activation_func : activation function, e.g. "nn.ReLU" (without parenthesis)
        
        name : name (str) of QNet when saved

        """
        super().__init__()
        
        self.checkpoint_file = os.path.join(checkpoint_dir, name+"_sac.pt")
        self.q = utils.mlp([num_obs + num_actions] + list(hidden_sizes) + [1], activation_func)
        
    def forward(self, observation, action):
        q = self.q(torch.cat([observation, action], dim=-1))
        return torch.squeeze(q, -1)
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))










