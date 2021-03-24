# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 18:50:50 2021

@author: groes
"""
import reinforcement_learning.task3.buffer as buff
import numpy as np 
import reinforcement_learning.task3.utils as utils

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

####################### hyperparams #########################

unittest_environment_params = {
    'N' : 10,
    'stumble_prob' : 0.3
    }

unittest_buffer_params = {
    'obs_dims': (25,4),
    'num_actions': 9
    }

unittest_ac_params = {
    'num_obs' : 13,
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

################### classes ##################################

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

unittest_DSAC = sac.DiscreteSAC(unittest_ac_params, unittest_params)

unittest_environment = bw.Environment(unittest_environment_params)
#################### unittests ###############################

def unittest_convert_obs():
    observation = unittest_environment.calculate_observations()
    print(observation['surroundings'])
    print(observation['relative_coordinates_car'])
    print(observation['relative_coordinates_britney'])
    converted_obs = utils.convert_state(observation)
    converted_obs = converted_obs.squeeze()
    print(converted_obs)
    some_zeros = torch.zeros(13)
    print(some_zeros[0])

unittest_convert_obs()


def unittest_actor_critic():
    observation = unittest_environment.calculate_observations()
    converted_obs = utils.convert_state(observation)
    converted_obs = converted_obs.squeeze()
    
    #converted_obs = torch.ones(13, dtype=torch.float32)
    test = torch.zeros(13)
    output = unittest_ac.policy(converted_obs)
    assert len(output) == unittest_ac_params['num_actions']
    
    output_q1 = unittest_ac.q1(converted_obs)
    assert len(output_q1) == unittest_ac_params['num_actions']
    
    output_q2 = unittest_ac.q2(converted_obs)
    assert len(output_q2) == unittest_ac_params['num_actions']
   

unittest_actor_critic()
    
    
def unittest_environment_step():
    unittest_environment.reset()
    len(unittest_buffer.reward_memory)
    unittest_DSAC.environment_step(unittest_environment, unittest_buffer)
    len(unittest_buffer.reward_memory)
 

def unittest_buffer():
    size = 100
    input_shape = (9, 4)
    num_actions = 10
    buffer = buff.ReplayBuffer(input_shape, num_actions, size)
    
    assert len(buffer) == 100 and buffer.memory_size == 100
    assert buffer.memory_counter == 0
        
    state = 10
    action = 1
    reward = -1
    new_state = 98
    done = 0
    
    buffer.append(state, action, reward, new_state, done)
    
    assert buffer.memory_counter == 1
    assert len(buffer) == size and buffer.memory_size == size
    assert buffer.action_memory[0].all() == action
    
    for i in np.round(buffer.state_memory[0][0], 2):
        assert i == round(state, 2)
        
    # Checking that unused index is still 0
    assert buffer.action_memory[1].all() == 0
   
unittest_buffer()


def unittest_convert_state():
    env_size = 5
    norm = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    surroundings = np.array([[1,1,1],
                             [0,0,1],
                             [0,0,1]])
    
    relative_coor_car = np.array([-1, 2])
    
    relative_coor_britney = np.array([4, 3])
    
    observations = {"surroundings" : surroundings,
                    "relative_coordinates_car" : relative_coor_car,
                    "relative_coordinates_britney" : relative_coor_britney}
    
    tensor = utils.convert_state(observations, env_size = env_size, norm=norm,
                                 device=device)
    
    assert tensor.shape[1] == surroundings.size + len(relative_coor_britney) + len(relative_coor_car)
    assert round(tensor[0][0].item(), 2) == -0.20
    assert round(tensor[0][1].item(), 2) == 0.40
    assert round(tensor[0][2].item(), 2) == 0.80
    assert round(tensor[0][-1].item(), 2) == 0.50
    
    
unittest_convert_state()