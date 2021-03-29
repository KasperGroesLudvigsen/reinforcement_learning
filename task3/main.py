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
    'clipping_norm': None,
    'tune_temperature' : 0.5,
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

environment = bw.Environment(environment_params)
environment.reset()
buffer = buf.ReplayBuffer(
    buffer_params['obs_dims'],
    buffer_params['num_actions']
    )
environment.display()


def learning_environment(number_of_episodes):

    #fill up buffer
    
    for _ in range(32):
        done = False
        while not done:
            done = DSAC.environment_step(environment, buffer)
    
    ran_out_of_time = 0
    success = 0
        
    for _ in range(number_of_episodes): 
        environment.respawn()
        
        
        
        #if params['batch_size'] > buffer.memory_counter:
        
        environment.respawn
        done = False
        while not done:
            done = DSAC.environment_step(environment, buffer)
            environment.display()
            DSAC.gradient_step_experiment(buffer, params['batch_size'])
            # add some visual stuff
        if environment.time_elapsed == environment.time_limit:
            ran_out_of_time += 1
        else:
            success +=1
    print("times ran out: {}".format(ran_out_of_time))
    print("successes: {}".format(success))       

learning_environment(1)
learning_environment(200)
#earning_environment(2000)
#SAC.alpha = 0.1


#for _ in range(10):
#    learning_environment(10)
#environment.display()
