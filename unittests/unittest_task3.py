# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 18:50:50 2021

@author: groes
"""
import task3.buffer as buff
import numpy as np 
import task3.utils as utils
import torch 

def unittest_buffer():
    size = 100
    input_shape = (9, 4)
    num_actions = 10
    buffer = buff.ReplayBuffer(input_shape, num_actions, size)
    
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