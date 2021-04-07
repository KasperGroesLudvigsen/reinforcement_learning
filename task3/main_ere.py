import numpy as np
from matplotlib import pyplot as plt

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
import task3.buffer_ere as buf
import task3.networks.q_network as qnet
import task3.Discrete_SAC as SAC

####################### hyperparams #########################

environment_params = {
    'N' : 10,
    'stumble_prob' : 0
    }

buffer_params = {
    'obs_dims': (1,125),
    'num_actions': 10
    }

ac_params = {
    'num_obs' : 125,
    'num_actions' : 10,
    'hidden_sizes' : [256,256],
    'activation_func': nn.ReLU
     }

params = {
    'lr': 0.0003,
    'gamma': 0.9,
    'batch_size': 64,
    'polyak' : 0.05,
    'clipping_norm': 1,
    "automatic_entropy_tuning":False,
    "entropy_alpha":0.3
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


def learning_environment(number_of_episodes, display = False):
    training_scores = []
    validation_scores = []
    #rewards = []
    #fill up buffer
    
    for _ in range(1000):
        done = False
        environment.reset()
        #environment.respawn()
        while not done:
            done = DSAC.environment_step(environment, buffer, buffer_fill=True)
    
    
    ran_out_of_time = 0
    success = 0
        
    for _ in range(number_of_episodes):
        print('Starting Episode: ', _)
        environment.reset()
        #environment.respawn()
        
        done = False
        rewardz = 0
        while not done:
            done, reward = DSAC.environment_step(environment, buffer, buffer_fill = False)
            rewardz += reward
            if display:
                environment.display()
            
            #DSAC.gradient_step_experiment(buffer, params['batch_size'])
            #DSAC.gradient_step_experiment(buffer, params['batch_size'])
            # add some visual stuff#
            
        big_k = 100
        little_k = 1
        for a in range(big_k):
            states, new_states, actions, rewards, dones = buffer.sample(params['batch_size'], big_k, little_k)
            DSAC.gradient_step_ere(states, new_states, actions, rewards, dones)
            little_k += 1
        
        training_scores.append(rewardz)    
        #training_scores.append(environment.time_elapsed)
        #rewards.append()
        if environment.time_elapsed == environment.time_limit:
            ran_out_of_time += 1
        else:
            success +=1
            
        if _ % 10 ==0:
            print('Strarting Valdiation loop')
            DSAC.train_mode = False
            environment.reset()
            #environment.respawn()
        
            done = False
            val_rewardz = 0
            while not done:
                done, reward = DSAC.environment_step(environment, buffer, buffer_fill = False)
                if display:
                    environment.display()
                val_rewardz += reward
            validation_scores.append(val_rewardz)
            DSAC.train_mode = True
                
            
    plt.plot(training_scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.grid(True)
    plt.show()
    
    plt.plot(validation_scores)
    plt.title('Validation Scores')
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.grid(True)
    plt.show()
    
    print("times ran out: {}".format(ran_out_of_time))
    print("successes: {}".format(success))       

#learning_environment(1)
#DSAC.alpha=0
#learning_environment(1000,0,1000)
learning_environment(21)
#DSAC.train_mode = False
#learning_environment(1000,0, 10000)
#SAC.alpha = 0.


#for _ in range(10):
#    learning_environment(10)
#environment.display()
