# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:39:48 2021

@author: groes
"""
import torch.nn as nn
import numpy as np
import torch

def run_episode(env, policy, stumble_probability, print_iter=False):
    total_reward = 0
    done = False
    iterations = 0
    while not done:
        iterations += 1
        observations, reward, done = env.take_action_guard(policy)
        total_reward += reward
        env.britney_stubmles(stumble_probability)
        # done = env.run(random_policy())
        if iterations % 10 == 0:
            env.display()
        if done == True and print_iter == True:
            print("It took {} iterations for Britney to reach the car".format(
                iterations))
    return iterations, total_reward



def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def convert_state(state, env_size, norm, device):
    # Adapted from lab 7 feedback
    """
    state (dict) : dictionary with relative coordinates and agent's surroundings
    
    env_size (int) : length of the environment's sides
    
    norm (float) : some float, e.g. 4.0, by which to normalize surroundings
    
    device (torch object) : GPU or CPU
    
    """
    rel_coor_car = state['relative_coordinates_car'].flatten()/env_size
    rel_coor_britney = state['relative_coordinates_britney'].flatten()/env_size
    o = state['surroundings'].flatten()/norm
    state_tensor = np.concatenate([rel_coor_car, rel_coor_britney, o])
    state_tensor = torch.tensor(state_tensor, device=device).unsqueeze(0)
    
    return state_tensor