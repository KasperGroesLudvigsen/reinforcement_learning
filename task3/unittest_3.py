# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 12:06:18 2021

@author: hugha
"""

import torch 
#import networks.q_network as qnet
import torch.nn.functional as F
#import networks.policy_network as pi_net
import reinforcement_learning.task3.Discrete_SAC as sac
import torch.nn as nn
from copy import deepcopy
import itertools
from torch.optim import Adam
import reinforcement_learning.task1.britneyworld as bw
import reinforcement_learning.task3.buffer as buf

unittest_environment_params = {
    'N' : 10,
    'stumble_prob' : 0.3
    }

unittest_buffer_params = {
    'obs_dims': 25,
    'num_actions': 9
    }

unittest_ac_params = {
    'num_obs' : 25,
    'num_actions' : 9,
    'hidden_sizes' : [100,100],
    'activation_func': nn.ReLU
    }

unittest_params = {
    'lr': 0.001,
    'alpha' : 0.1,
    'gamma': 0.9,
    'batch_size': 32
    }


unittest_ac = sac.Actor_Critic(
    unittest_ac_params['num_obs'],
    unittest_ac_params['num_actions'], 
    unittest_ac_params['hidden_sizes'],
    unittest_ac_params['activation_func']
    )

unittest_buffer = buf.ReplayBuffer(
    unittest_buffer_params['obs_dims'],
    unittest_buffer_params['num_actions']
    )

def unittest_actor_critic():
    unittest_observation = torch.zeros(25)
    
    output = unittest_ac.policy(unittest_observation)
    assert len(output) == unittest_ac_params['num_actions']
    
    output_q1 = unittest_ac.q1(unittest_observation)
    assert len(output_q1) == unittest_ac_params['num_actions']
    
    output_q2 = unittest_ac.q2(unittest_observation)
    assert len(output_q2) == unittest_ac_params['num_actions']
   

unittest_actor_critic()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    