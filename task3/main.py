import torch 
import networks.q_network as qnet
import torch.nn.functional as F
import networks.policy_network as pi_net
import networks.Discrete_SAC as sac
import torch.nn as nn
from copy import deepcopy
import itertools
from torch.optim import Adam
import reinforcement_learning.task1.britneyworld as bw
import reinforcement_learning.task3.buffer as buf

environment_params = {
    'N' : 10,
    'stumble_prob' : 0.3
    }

buffer_params = {
    'obs_dims': 9,
    'number_of_actions': 9
    }

ac_params = {
    'num_obs' : 25,
    'num_actions' : 9,
    'hidden_sizes' : [100,100],
    'activation_func': 'ReLu'
    }

params = {
    'lr': 0.001,
    'alpha' : 0.1,
    'gamma': 0.9,
    'batch_size': 32
    }

from reinforcement_learning.task3.Discrete_SAC import DiscreteSAC 
ds = DiscreteSAC(ac_params, params)
environment = bw.Environment(environment_params)
buffer = buf.ReplayBuffer(buffer_params['obs_dims'], buffer_params['number_of_actions'])



def learning_environment(number_of_episodes):
    softac = sac.Actor_Critic(params)
    environment = bw.britneyworld(environment_params)
    buffer = buf.ReplayBuffer(buffer_params)
    
    for _ in number_of_episodes: 
        environment.respawn
        Done = False
        while not Done:
            softac.environment_step(environment, buffer)
            softac.gradient_step(buffer, params['batch_size'])
        
        # add some visual stuff
    