import numpy as np 

import task3.utils as utils

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
    'stumble_prob' : 1
    }

buffer_params = {
    'obs_dims': (1,6),
    'num_actions': 10
    }

ac_params = {
    'num_obs' : 6,
    'num_actions' : 10,
    'hidden_sizes' : [32,32],
    'activation_func': nn.ReLU
    }

params = {
    'lr': 0.0001,
    'alpha' : 0.99,
    'gamma': 0.99,
    'batch_size': 256,
    'polyak' : 0.5,
    'clipping_norm': 2,
    'tune_temperature' : 0.5,
    "automatic_entropy_tuning":False,
    "entropy_alpha":0.0
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


def learning_environment(seed, random_episodes, number_of_episodes):

    #fill up buffer
    
    for _ in range(seed):
        done = False
        environment.reset()
        while not done:
            done = DSAC.environment_step(environment, buffer, buffer_fill=True)
    
    
    for _ in range(random_episodes):
        done = False
        environment.reset()
        while not done:
            done = DSAC.environment_step(environment, buffer, buffer_fill=True)
            DSAC.gradient_step_experiment(buffer, params['batch_size'])
            #DSAC.gradient_step_experiment(buffer, params['batch_size'])
            #DSAC.gradient_step_experiment(buffer, params['batch_size'])
    ran_out_of_time = 0
    success = 0
        
    for _ in range(number_of_episodes): 
        environment.reset()
        
        done = False
        while not done:
            done = DSAC.environment_step(environment, buffer, buffer_fill = False)
            #environment.display()
            #DSAC.gradient_step_experiment(buffer, params['batch_size'])
            DSAC.gradient_step_experiment(buffer, params['batch_size'])
            #DSAC.gradient_step_experiment(buffer, params['batch_size'])
            # add some visual stuff
        if environment.time_elapsed == environment.time_limit:
            ran_out_of_time += 1
        else:
            success +=1
    print("times ran out: {}".format(ran_out_of_time))
    print("successes: {}".format(success))       

#learning_environment(1)
#learning_environment(1000, 200)
learning_environment(0,0, 1)
#SAC.alpha = 0.


#for _ in range(10):
#    learning_environment(10)
#environment.display()
