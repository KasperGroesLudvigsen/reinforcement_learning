# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:13:13 2021

@author: groes
"""

import task1.britneyworld as bw
import random 
import numpy as np
import task2.qlearning as ql
import task2.policies as pol
import task2.task2_utils as utils
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import pandas as pd
import task2.early_stopping as early_stopping
from scipy import stats
import seaborn as sns

runs = 3
env_size = 20
episodes = 10000
epsilon = 0.1
epsilon_decay = 0.99
discount_rate = 0.9 # gamma
learning_rate = 0.1 # alpha
num_actors = 2
stumble_prob = 0.2 # 0.2


success_episodes = []

env = bw.Environment(env_size, stumble_prob)
env.reset()
policy = pol.E_Greedy_Policy(epsilon, epsilon_decay, env.guard_actions)
algo = ql.QLearning(env, discount_rate, num_actors, learning_rate)
env.guard_location

rewards = []
for run in range(runs):

    
    for i in range(episodes):
        #print(f"Starting episode {i}")
        total_reward = 0
        done = False
        
        
        
        while not done:
            state = env.get_state()
            action = policy(state, algo.q_table)
            action_idx = env.guard_actions.index(action)
            #env.display()
            new_state, reward, done = env.take_action_guard(
                env.guard_location, env.britney_location, action)
            total_reward += reward
            
            if not done:
                
                max_q_next = np.max(algo.q_table[new_state])
                
                current_q = algo.q_table[state][action_idx]
                
                new_q = (1-learning_rate) * current_q + learning_rate * (reward + discount_rate * max_q_next)
                
                algo.q_table[state][action_idx] = new_q
                
                
                
        rewards.append(total_reward)
        
        env.respawn()
        
        
        
        
    successful_episodes = 0
    for i in rewards:
        if i > 0:
            successful_episodes += 1
    success_episodes.append(successful_episodes)
    print(f"Successful episodes {successful_episodes}")