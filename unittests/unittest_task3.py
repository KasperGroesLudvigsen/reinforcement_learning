# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 18:50:50 2021

@author: groes
"""
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


unittest_environment_params = {
    'N' : 10,
    'stumble_prob' : 0.4
    }

unittest_buffer_params = {
    'obs_dims': (1,6),
    'num_actions': 10
    }

unittest_ac_params = {
    'num_obs' : 6,
    'num_actions' : 10,
    'hidden_sizes' : [100,100],
    'activation_func': nn.ReLU
    }

unittest_params = {
    'lr': 0.001,
    'alpha' : 0.1,
    'gamma': 0.9,
    'batch_size': 32,
    'polyak' : 0.8,
    'clipping_norm': None,
    'tune_temperature' : 0.3,
    "automatic_entropy_tuning":False,
    "entropy_alpha":0.0
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
unittest_environment.reset()
#################### unittests ###############################

def unittest_convert_obs():
    observation = unittest_environment.calculate_observations()
    print(observation['surroundings'])
    print(observation['relative_coordinates_car'])
    print(observation['relative_coordinates_britney'])
    converted_obs = utils.convert_state(observation)
    converted_obs = converted_obs.squeeze()
    print(converted_obs)
    #some_zeros = torch.zeros(13)
    #print(some_zeros[0])

unittest_convert_obs()


def unittest_actor_critic():
    unittest_environment.reset()
    observation = unittest_environment.calculate_observations()
    converted_obs = utils.convert_state(observation)
    print(converted_obs)
    converted_obs = converted_obs.squeeze()
    converted_obs = torch.tensor([0.5,0.5, 0.6, 0.7, 0.1, 0.1])
    with torch.no_grad():
        output = unittest_ac.policy(converted_obs)
    #assert len(output) == unittest_ac_params['num_actions']
    action = np.argmax(np.array(output))
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
    observation = unittest_environment.calculate_observations()
    converted_obs = utils.convert_state(observation)
    print(converted_obs)
    hughs_unittest_buffer = buf.ReplayBuffer(
        unittest_buffer_params['obs_dims'],
        unittest_buffer_params['num_actions'], 
        memory_size=100
        )
    
    for _ in range(50):
        unittest_DSAC.environment_step(unittest_environment, hughs_unittest_buffer, buffer_fill = False)
    
    #print(len(unittest_buffer.reward_memory))
    #unittest_DSAC.environment_step(unittest_environment, unittest_buffer, buffer_fill=False)
    observation = unittest_environment.calculate_observations()
    converted_obs = utils.convert_state(observation)
    print(converted_obs)
    
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
    
    hughs_unittest_buffer = buf.ReplayBuffer(
        unittest_buffer_params['obs_dims'],
        unittest_buffer_params['num_actions'], 
        memory_size=100
        )
    
    for _ in range(50):
        unittest_DSAC.environment_step(unittest_environment, hughs_unittest_buffer, buffer_fill = False)
    
    states, new_states, actions, rewards, dones = hughs_unittest_buffer.sample(30)
    assert len(states) == 30
    assert len(new_states) == 30
    assert len(actions) == 30
    assert len(rewards) == 30
    assert len(dones) == 30
    
    #q1 = unittest_DSAC.actor_critic.q1(states.squeeze()) 
    #print("q1 before gathering{}:".format(q1))
    #q1 = q1.gather(1, actions.long())
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
    original_policy_model = deepcopy(unittest_DSAC.actor_critic.policy)
    hughs_unittest_buffer = buf.ReplayBuffer(
        unittest_buffer_params['obs_dims'],
        unittest_buffer_params['num_actions'],
        memory_size= 100
        )

    for _ in range(50):
        unittest_DSAC.environment_step(unittest_environment, hughs_unittest_buffer, buffer_fill = False)

    states, new_states, actions, rewards, dones = hughs_unittest_buffer.sample(30)
    
    unittest_DSAC.pi_optimizer.zero_grad()
    policy_loss = unittest_DSAC.calc_policy_loss(states)
    print(policy_loss)
    policy_loss.backward()
    unittest_DSAC.pi_optimizer.step()

    for new_param, original_param in zip(unittest_DSAC.actor_critic.policy.parameters(), original_policy_model.parameters()):
        print("new_param",new_param)
        print("original param", original_param)

unittest_calculate_policy_loss()


def unittest_calculate_q_loss():
    unittest_environment.reset()
    
    original_q1_model = deepcopy(unittest_DSAC.actor_critic.q1)
    original_q2_model = deepcopy(unittest_DSAC.actor_critic.q2)
    original_policy_model = deepcopy(unittest_DSAC.actor_critic.policy)
    
    hughs_unittest_buffer = buf.ReplayBuffer(
        unittest_buffer_params['obs_dims'],
        unittest_buffer_params['num_actions'],
        memory_size= 100
        )
    
    for _ in range(50):
        unittest_DSAC.environment_step(unittest_environment, hughs_unittest_buffer, buffer_fill=False)

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
    unittest_DSAC.q1_optimizer.zero_grad()
    unittest_DSAC.q2_optimizer.zero_grad()
    q1_loss, q2_loss = unittest_DSAC.calc_q_loss(states, actions, rewards, new_states, dones)
    q1_loss.backward()
    q2_loss.backward()
    unittest_DSAC.q1_optimizer.step()
    unittest_DSAC.q2_optimizer.step()
    for new_param, original_param in zip(unittest_DSAC.actor_critic.q2.parameters(), original_q2_model.parameters()):
        print("new_param",new_param)
        print("original param", original_param)

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


unittest_take_optimization_step()



def unittest_gradient():
    unittest_environment.reset()

    hughs_unittest_buffer = buf.ReplayBuffer(
        unittest_buffer_params['obs_dims'],
        unittest_buffer_params['num_actions'],
        memory_size= 100
        )
    
    for _ in range(50):
        dont = unittest_DSAC.environment_step(unittest_environment, hughs_unittest_buffer,buffer_fill = False)
    
    unittest_DSAC.gradient_step(hughs_unittest_buffer, 30)

unittest_gradient()



    
    
    
