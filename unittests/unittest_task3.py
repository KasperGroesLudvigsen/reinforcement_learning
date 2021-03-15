# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 18:50:50 2021

@author: groes
"""
import task3.buffer as buff
import numpy as np 

def unittest_buffer():
    size = 100
    shape = (9, 4)
    num_actions = 10
    buffer = buff.ReplayBuffer(size, shape, num_actions)
    
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
    