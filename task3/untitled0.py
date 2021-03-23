# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:12:34 2021

@author: hugha
"""

from reinforcement_learning.task3.Discrete_SAC import Actor_Critic
from reinforcement_learning.task3.Discrete_SAC import DiscreteSAC 
import torch.nn as nn
import torch

actions = ["N", "E", "S", "W", "NE", "SE", "SW", "NW" ,"push"]

ut_num_obs = 25
num_actions = len(actions)
hidden_sizes = [100,100]
activation_func = nn.ReLU
ac = Actor_Critic(num_obs, num_actions, hidden_sizes, activation_func)


observation = torch.zeros(25)
output = ac.policy(observation)
output_q1 = ac.q1(observation)
output_q2 = ac.q2(observation)

print(output)
print(output_q1)
print(output_q2)

ac_params = {
    'num_obs' : 25,
    'num_actions' : 9,
    'hidden_sizes' : [100,100],
    'activation_func': nn.ReLU
    }

params = {
    'lr': 0.001,
    'alpha' : 0.1,
    'gamma': 0.9,
    'batch_size': 32
    }



ds = DiscreteSAC(ac_params, params)