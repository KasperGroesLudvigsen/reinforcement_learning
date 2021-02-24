# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:03:47 2021

@author: groes
"""

from task1 import britneyworld as bw
import numpy as np

def unittest_get_next_position():
    env = bw.Environment(10)
    env.reset()
    env.map = np.array([[1,2,3],
                        [3,4,6],
                        [7,8,9]])
    next_position = env.get_next_position("N", np.array([1,1]))
    assert env.map[next_position[0], next_position[1]] == 2
    next_position = env.get_next_position("S", np.array([1,1]))
    assert env.map[next_position[0], next_position[1]] == 8
    next_position = env.get_next_position("E", np.array([1,1]))
    assert env.map[next_position[0], next_position[1]] == 6
    next_position = env.get_next_position("W", np.array([1,1]))
    assert env.map[next_position[0], next_position[1]] == 3
    next_position = env.get_next_position("NE", np.array([1,1]))
    assert env.map[next_position[0], next_position[1]] == 3
    next_position = env.get_next_position("SE", np.array([1,1]))
    assert env.map[next_position[0], next_position[1]] == 9
    next_position = env.get_next_position("SW", np.array([1,1]))
    assert env.map[next_position[0], next_position[1]] == 7
    next_position = env.get_next_position("NW", np.array([1,1]))
    assert env.map[next_position[0], next_position[1]] == 1
    
unittest_get_next_position()  





    
def unittest_any_grid_size():
    pass
    # test that the grid can be initialized with any size