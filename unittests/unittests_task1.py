# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:03:47 2021

@author: groes
"""

#from task1 import britneyworld as bw
import task1.britneyworld as bw
import task1.assassinworld as r1
import numpy as np


def unittest_get_next_position():
    env = bw.Environment(10, 0.2)
    env.reset()
    env.map = np.array([[0,0,0],
                        [0,0,0],
                        [0,0,0]])
    
    next_position = env.get_next_position("N", np.array([1,1]))
    assert next_position[0] == 0 and next_position[1] == 1
    
    next_position = env.get_next_position("S", np.array([1,1]))
    assert next_position[0] == 2 and next_position[1] == 1
    
    next_position = env.get_next_position("E", np.array([1,1]))
    assert next_position[0] == 1 and next_position[1] == 2
    
    next_position = env.get_next_position("W", np.array([1,1]))
    assert next_position[0] == 1 and next_position[1] == 0
    
    next_position = env.get_next_position("NE", np.array([1,1]))
    assert next_position[0] == 0 and next_position[1] == 2
    
    next_position = env.get_next_position("SE", np.array([1,1]))
    assert next_position[0] == 2 and next_position[1] == 2
    
    next_position = env.get_next_position("SW", np.array([1,1]))
    assert next_position[0] == 2 and next_position[1] == 0
    
    next_position = env.get_next_position("NW", np.array([1,1]))
    assert next_position[0] == 0 and next_position[1] == 0
    
unittest_get_next_position()  


def unittest_any_grid_size():
    env = bw.Environment(4, 0.1)
    env.reset()
    env = bw.Environment(5, 0.1)
    env.reset()
    env = bw.Environment(50, 0.1)
    env.reset()    
    env = bw.Environment(100, 0.1)
    env.reset()
    env = bw.Environment(104, 0.1)
    env.reset()
    env = bw.Environment(263, 0.1)
    env.reset()
    env = bw.Environment(1000, 0.1)
    env.reset()
  
unittest_any_grid_size()

def unittest_push_britney():
    env = bw.Environment(10, 0.0)
    env.reset()
    env.britney_location = np.array([3,3])
    env.guard_location = np.array([3,4])    
    env.push_britney()
    
    assert env.britney_location[0] == 3
    assert env.britney_location[1] == 2
    
unittest_push_britney()


def unittest_are_locations_adjacent():
    env = bw.Environment(10, 0.1)
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

"""
def unittest_move_ass():

    britney_loc = np.array([3,3])
    ass_loc = np.array([[1,1]])
    ass_env = r1.AssassinWorld(10, 0.1, ass_loc)
    
    ass_new_loc = ass_env.move_assassins(britney_loc, ass_loc)
    
    assert ass_new_loc[0][0] == 2.0 and ass_new_loc[0][1] == 2.0
    
    britney_loc = np.array([8,8])
    ass_loc = np.array([[1,3], [5,8], [1,9]])
    
    ass_new_loc = ass_env.move_assassins(britney_loc, ass_loc)
    
    assert (ass_new_loc[0] == np.array([2,4])).all()
    assert (ass_new_loc[1] == np.array([6,8])).all()
    assert (ass_new_loc[2] == np.array([2,8])).all()

unittest_move_ass()

"""








