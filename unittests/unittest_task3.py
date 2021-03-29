# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 18:50:50 2021

@author: groes
"""
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


unittest_environment_params = {
    'N' : 10,
    'stumble_prob' : 0.3
    }

unittest_buffer_params = {
    'obs_dims': (1,13),
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
    'batch_size': 32,
    'polyak' : 0.8,
    'clipping_norm': 0.7,
    'tune_temperature' : 0.3,
    "automatic_entropy_tuning":False,
    "entropy_alpha":0.5
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
    print(converted_obs.shape)
    some_zeros = torch.zeros(13)
    print(some_zeros[0])

unittest_convert_obs()


def unittest_actor_critic():
    unittest_environment.reset()
    observation = unittest_environment.calculate_observations()
    converted_obs = utils.convert_state(observation)
    converted_obs = converted_obs.squeeze()
    with torch.no_grad():
        output = unittest_ac.policy(converted_obs)
    assert len(output) == unittest_ac_params['num_actions']
    action = np.max(np.array(output))
    print(action)
    print(np.array(output))
    output_q1 = unittest_ac.q1(converted_obs)
    assert len(output_q1) == unittest_ac_params['num_actions']
    #print(output_q1)
    output_q2 = unittest_ac.q2(converted_obs)
    assert len(output_q2) == unittest_ac_params['num_actions']
    #print(output_q2)
    
    _, reward, done = unittest_environment.take_action_guard(
            unittest_environment.guard_location,
            unittest_environment.britney_location,
            action)

unittest_actor_critic()
    
    
def unittest_environment_step():
    unittest_environment.reset()
    #print(len(unittest_buffer.reward_memory))
    unittest_DSAC.environment_step(unittest_environment, unittest_buffer)
    #len(unittest_buffer.reward_memory)
    print(unittest_buffer.reward_memory[0])
    
unittest_environment_step()


def unittest_buffer():
    size = 100
    input_shape = (9, 4)
    num_actions = 10
    buffer = buf.ReplayBuffer(input_shape, num_actions, size)
    
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

def hughs_much_better_unittest_buffer():
    unittest_environment.reset()
    
    unittest_buffer = buf.ReplayBuffer(
        unittest_buffer_params['obs_dims'],
        unittest_buffer_params['num_actions'], 
        memory_size=100
        )
    
    for _ in range(50):
        unittest_DSAC.environment_step(unittest_environment, hughs_unittest_buffer)
    
    states, new_states, actions, rewards, dones = hughs_unittest_buffer.sample(30)
    assert len(states) == 30
    assert len(new_states) == 30
    assert len(actions) == 30
    assert len(rewards) == 30
    assert len(dones) == 30
    
    q1 = unittest_DSAC.actor_critic.q1(states.squeeze()) 
    #print("q1 before gathering{}:".format(q1))
    q1 = q1.gather(1, actions.long())
    #print("q1 after gathering{}:".format(q1[:,0]))
    #print(states.squeeze())
    #print(new_states.shape)
    #print(actions)
    #print(actions)
    print(dones)
    
hughs_much_better_unittest_buffer()
    
  
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


def unittest_calculate_policy_loss():
    unittest_environment.reset()

    hughs_unittest_buffer = buf.ReplayBuffer(
        unittest_buffer_params['obs_dims'],
        unittest_buffer_params['num_actions'],
        memory_size= 100
        )

    for _ in range(50):
        unittest_DSAC.environment_step(unittest_environment, hughs_unittest_buffer)

    states, new_states, actions, rewards, dones = hughs_unittest_buffer.sample(30)


    policy_loss = unittest_DSAC.calc_policy_loss(states)


    print(policy_loss)

unittest_calculate_policy_loss()


def unittest_calculate_q_loss():
    unittest_environment.reset()

    hughs_unittest_buffer = buf.ReplayBuffer(
        unittest_buffer_params['obs_dims'],
        unittest_buffer_params['num_actions'],
        memory_size= 100
        )
    
    for _ in range(50):
        unittest_DSAC.environment_step(unittest_environment, hughs_unittest_buffer)

    states, new_states, actions, rewards, dones = hughs_unittest_buffer.sample(30)    
    #print(states.squeeze().shape)
    #print(new_states.shape)
    #print(actions.shape)
    states = states.squeeze()
    new_states = new_states.squeeze()
    #q1 = unittest_ac.policy(new_states.squeeze())
    #q2 = unittest_ac.policy(new_states)
    #print(q1.shape)
    #print(q2.shape)
    q1_loss, q2_loss = unittest_DSAC.calc_q_loss(states, actions, rewards, new_states, dones)

    print(q1_loss)
    print(q2_loss)

unittest_calculate_q_loss()


def unittest_take_optimization_step():
    dsac = sac.DiscreteSAC(unittest_ac_params, unittest_params)

    # Will compare against copy to see if model params have changed after         
    original_q1_model = deepcopy(dsac.actor_critic.q1)
    original_q2_model = deepcopy(dsac.actor_critic.q2)
    original_policy_model = deepcopy(dsac.actor_critic.policy)

    # Taking 10 optimization steps and checking after each 
    for _ in range(10):

        # Mocking loss
        mse = nn.MSELoss()
        input = torch.randn(3, 5, requires_grad=True)
        target = torch.randn(3, 5)
        mock_loss = mse(input, target)

        # Taking optimization step
        dsac.take_optimization_step(
            optimizer=dsac.q1_optimizer,
            network=dsac.actor_critic.q1,
            loss=mock_loss
            )

        # Comparing original optimizer params with params after step
        for new_param, original_param in zip(dsac.actor_critic.q1.parameters(), original_q1_model.parameters()):
            print(new_param)
            print(original_param)
            if len(new_param) > 1:
                for old_tensor, new_tensor in zip(original_param, new_param):
                    equal = all(new_tensor.eq(old_tensor))
                    assert not equal
            equal = all(new_param.eq(original_param))
            assert not equal


    ###############################################################
    # Below is garbage
    """
    dsac_ac_copy = deepcopy(dsac.actor_critic)
    for p in dsac_ac_copy.parameters():
        p.requires_grad = False
    q1_state_dict = dsac_ac_copy.q1.state_dict()
    q2_state_dict = dsac_ac_copy.q2.state_dict()
    policy_state_dict = dsac_ac_copy.policy.state_dict()

    # Creating loss
    #mse = nn.MSELoss()
    #input = torch.randn(3, 5, requires_grad=True)
    #target = torch.randn(3, 5)
    #loss = mse(input, target)


    for _ in range(50):
        unittest_DSAC.environment_step(unittest_environment, unittest_buffer)

    states, new_states, actions, rewards, dones = unittest_buffer.sample(30)    
    states = states.squeeze()
    new_states = new_states.squeeze()
    q1_loss, q2_loss = unittest_DSAC.calc_q_loss(states, actions, rewards, new_states, dones)



    # For easy looping over keys in dict
    dict_keys = list(q1_state_dict.keys())

    # Taking optimization step for all three nets / optimizers
    # Ain't no half stepping
    for _ in range(1000):

        states, new_states, actions, rewards, dones = unittest_buffer.sample(30)    
        states = states.squeeze()
        new_states = new_states.squeeze()
        q1_loss, q2_loss = unittest_DSAC.calc_q_loss(states, actions, rewards, new_states, dones)

        unittest_DSAC.take_optimization_step(
            optimizer=unittest_DSAC.q1_optimizer,
            network=unittest_DSAC.actor_critic.q1,
            loss=q1_loss
            )

    #unittest_DSAC.take_optimization_step(
    #    optimizer=unittest_DSAC.q2_optimizer,
    #    network=unittest_ac.q2,
    #    loss=loss
    #    )

    #unittest_DSAC.take_optimization_step(
    #    optimizer=unittest_DSAC.pi_optimizer,
    #    network=unittest_ac.policy,
    #    loss=loss
    #    )

    dsac.q1_optimizer.param_groups


    new_q1_state_dict = dsac.actor_critic.q1.state_dict()
    for key in dict_keys:
        new_tensor = new_q1_state_dict[key]
        original_tensor = q1_state_dict[key]
        print(key)

        assert len(new_tensor) == len(original_tensor)


        for i in range(len(new_tensor)):
            a = new_tensor[i]
            b = original_tensor[i]
            equal = all(a.eq(b))
            if equal:

                print(i)

            #assert not equal

    print("HERE")
    # Asserting that parameters are not the same after step
    for key in range(len(dict_keys)):

        for i in range(len(q1_state_dict[dict_keys[key]])):   
            a = q1_state_dict[dict_keys[key]][i]
            b = unittest_DSAC.actor_critic.q1.state_dict()[dict_keys[key]][i]
            equal = all(a.eq(b))
            print(equal)
            assert not equal

        for i in len(q2_state_dict[dict_keys[key]]):   
            a = q2_state_dict[dict_keys[i]]
            b = unittest_DSAC.actor_critic.q2.state_dict()[dict_keys[key]][i]
            equal = all(a.eq(b))
            #assert not equal

        for i in len(policy_state_dict[dict_keys[key]]):   
            a = policy_state_dict[dict_keys[i]]
            b = unittest_DSAC.actor_critic.policy.state_dict()[dict_keys[key]][i]
            equal = all(a.eq(b))
            #assert not equal
    """
unittest_take_optimization_step()
  