import numpy as np 

import utils.utils as utils

#import reinforcement_learning.utils.utils as utils


import torch 
#import networks.q_network as qnet
import torch.nn.functional as F
#import networks.policy_network as pi_net
import task3.Discrete_SAC as sac
import torch.nn as nn
from copy import deepcopy
import itertools
from torch.optim import Adam
import task1.britneyworld as bw
import task3.buffer as buf
import task3.networks.q_network as qnet
import task3.Discrete_SAC as SAC

####################### hyperparams #########################

environment_params = {
    'N' : 10,
    'stumble_prob' : 0.3
    }

buffer_params = {
    'obs_dims': (1,13),
    'num_actions': 9
    }

ac_params = {
    'num_obs' : 13,
    'num_actions' : 9,
    'hidden_sizes' : [100,100],
    'activation_func': nn.ReLU
    }

params = {
    'lr': 0.001,
    'alpha' : 0.1,
    'gamma': 0.9,
    'batch_size': 32,
    'polyak' : 0.8,
    'clipping_norm': 0.7,
    'tune_temperature' : 0.3,
    "automatic_entropy_tuning":False,
    "entropy_alpha":0.5
    }

################# params ######################

ac = sac.Actor_Critic(
    ac_params['num_obs'],
    ac_params['num_actions'], 
    ac_params['hidden_sizes'],
    ac_params['activation_func']
    )

buffer = buf.ReplayBuffer(
    buffer_params['obs_dims'],
    buffer_params['num_actions']
    )

DSAC = sac.DiscreteSAC(ac_params, params)

#environment = bw.Environment(environment_params)





def learning_environment(number_of_episodes):

    environment = bw.Environment(environment_params)
    
    buffer = buf.ReplayBuffer(
        buffer_params['obs_dims'],
        buffer_params['num_actions']
        )
    
    for _ in range(number_of_episodes): 
        environment.respawn
        
        
        
        DSAC.environment_step(environment, buffer)
        #if params['batch_size'] > buffer.memory_counter:
        DSAC.gradient_step(buffer, params['batch_size'])
        
        # add some visual stuff

learning_environment(5)
