# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 16:01:46 2021

@author: groes
"""
import numpy as np
import torch

class ReplayBuffer:
    # Adapted from: 
    # https://www.youtube.com/watch?v=ioidsRlf79o&ab_channel=MachineLearningwithPhil
    def __init__(self, input_shape, num_actions, memory_size=1000000):
        """
        memory_size : max size of memory
        input_shape : shape/dimensions of observation from environment
        num_actions : number of actions available to the agent
        
        """
        self.memory_size = memory_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, *input_shape))
        # Store states that occur after actions
        self.new_state_memory = np.zeros((self.memory_size, *input_shape))
        self.action_memory = np.zeros((self.memory_size), dtype=np.float32)
        self.reward_memory = np.zeros((self.memory_size))
        # Store whether the state was terminal
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.int8)

    def __len__(self):
        return len(self.reward_memory)
    
    def append(self, state, action, reward, new_state, done):
        
        idx = self.memory_counter % self.memory_size
        
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.new_state_memory[idx] = new_state
        self.terminal_memory[idx] = done
        
        self.memory_counter += 1
        
    def sample(self, batch_size, big_k, little_k):
        
        
        
        n_stored_memories = min(self.memory_counter, self.memory_size)
        c_k = n_stored_memories*0.996**(1000*little_k/big_k)
        
        sample_range = range(int(c_k), n_stored_memories )
        
        #print(c_k, n_stored_memories)
        
        
        big_as_batch_can_be = min(int(n_stored_memories-c_k), batch_size)

        batch_idx = np.random.choice(sample_range, big_as_batch_can_be, replace=False)
        
        states = torch.as_tensor(self.state_memory[batch_idx], dtype=torch.float32)
        new_states = torch.as_tensor(self.new_state_memory[batch_idx], dtype=torch.float32)
        actions = torch.as_tensor(self.action_memory[batch_idx], dtype=torch.float32)
        rewards = torch.as_tensor(self.reward_memory[batch_idx], dtype=torch.float32)
        dones = torch.as_tensor(self.terminal_memory[batch_idx], dtype=torch.float32)
        
        states = states.squeeze()
        new_states = new_states.squeeze()
        
        return states, new_states, actions, rewards, dones
        
        
   