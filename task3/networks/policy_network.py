# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:36:26 2021

@author: groes
"""
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import utils.utils as utils

class PiNet(nn.Module):
    """
        This class is the actor network. It is the parametrised policy.
    """
    
    def __init__(self, num_obs, num_actions, hidden_sizes, activation_func, 
                 name, checkpoint_dir = "tmp/sac"):

        super().__init__()
        
        layer_sizes = [num_obs] + list(hidden_sizes) + [num_actions]
        self.q = utils.mlp(sizes=layer_sizes, activation=activation_func)
        self.checkpoint_file = os.path.join(checkpoint_dir, name+"_sac.pt")
        
    def forward(self, observation):
        q = self.q(observation)
        q = F.softmax(q, dim=1)
        q = q.squeeze(-1)
        return q
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        