# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:03:47 2021

@author: groes
"""

#from task1 import britneyworld as bw
import task1.britneyworld as bw
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
    env = bw.Environment(4)
    env.reset()
    env = bw.Environment(5)
    env.reset()
    env = bw.Environment(50)
    env.reset()    
    env = bw.Environment(100)
    env.reset()
    env = bw.Environment(104)
    env.reset()
    env = bw.Environment(263)
    env.reset()
    env = bw.Environment(1000)
    env.reset()
  
unittest_any_grid_size()

def unittest_push_britney():
    env = bw.Environment(10)
    env.reset()
    env.britney_location = np.array([3,3])
    env.guard_location = np.array([3,4])    
    env.push_britney()
    
    assert env.britney_location[0] == 3
    assert env.britney_location[1] == 2
    
unittest_push_britney()


def unittest_are_locations_adjacent():
    env = bw.Environment(10)
    env.reset()
    
    env.britney_location = np.array([3,3])
    env.guard_location = np.array([3,4])
    adjacent = env.are_locations_adjacent(env.britney_location, env.guard_location)
    assert adjacent

    env.britney_location = np.array([9,9])
    env.guard_location = np.array([8,8])
    adjacent = env.are_locations_adjacent(env.britney_location, env.guard_location)
    assert adjacent
    
    env.britney_location = np.array([3,3])
    env.guard_location = np.array([3,9])
    adjacent = env.are_locations_adjacent(env.britney_location, env.guard_location)
    assert not adjacent

    env.britney_location = np.array([0,3])
    env.guard_location = np.array([3,0])
    adjacent = env.are_locations_adjacent(env.britney_location, env.guard_location)
    assert not adjacent

unittest_are_locations_adjacent()













