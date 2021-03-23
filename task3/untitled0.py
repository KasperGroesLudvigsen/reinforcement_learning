# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:12:34 2021

@author: hugha
"""

from reinforcement_learning.task3.Discrete_SAC import Actor_Critic
import torch.nn as nn

actions = ["N", "E", "S", "W", "NE", "SE", "SW", "NW" ,"push"]

num_obs = 25
num_actions = len(actions)
hidden_sizes = [100,100]
activation_func = nn.ReLU
ac = Actor_Critic(num_obs, num_actions, hidden_sizes, activation_func)