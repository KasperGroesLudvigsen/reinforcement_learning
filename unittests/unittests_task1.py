# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:03:47 2021

@author: groes
"""

#from task1 import britneyworld as bw
import task1.britneyworld as bw
import task1.assassinworld as r1
import numpy as np
import math

def unittest_get_next_position():
    env = bw.Environment(10, 0.2)
    env.reset()
    env.map = np.array([[0,0,0],
                        [0,0,0],
                        [0,0,0]])
    
    # Setting these locations in order to be unaffected by is_empty()
    env.britney_location = np.array([1,1])
    env.guard_location = np.array([1,1])
    
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
    env = bw.Environment(104, 0.1)
    env.reset()

unittest_any_grid_size()

def unittest_push_britney():
    env = bw.Environment(10, 0.0)
    env.reset()
    env.britney_location = np.array([3,3])
    env.guard_location = np.array([3,4])    
    briney_loc = env.push_britney(env.britney_location, env.guard_location)
    
    assert briney_loc[0] == 3
    assert briney_loc[1] == 2
    
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


def unittest_pull_britney():
    env_size = 10
    env = bw.Environment(env_size, 0.2)
    env.reset()
    
    test_loc_britney = np.array([0, 0])
    test_loc_guard = np.array([1, 1])
    
    env.britney_location = test_loc_britney.copy()
    env.guard_location = test_loc_guard.copy()
    
    britney_loc, guard_loc = env.pull_britney(env.britney_location,
                                              env.guard_location)
    
    assert np.array_equal(britney_loc, test_loc_guard)
    assert np.array_equal(guard_loc, test_loc_guard)
    
    test_loc_britney = np.array([0, 0])
    test_loc_guard = np.array([0, 1])
    
    env.britney_location = test_loc_britney.copy()
    env.guard_location = test_loc_guard.copy()
    
    britney_loc, guard_loc = env.pull_britney(env.britney_location,
                                              env.guard_location)
    
    assert np.array_equal(britney_loc, test_loc_guard)
    assert np.array_equal(guard_loc, test_loc_guard)
    
    test_loc_britney = np.array([env_size-2, env_size-2])
    test_loc_guard = np.array([env_size-3, env_size-3])
    
    env.britney_location = test_loc_britney.copy()
    env.guard_location = test_loc_guard.copy()
    
    britney_loc, guard_loc = env.pull_britney(env.britney_location,
                                              env.guard_location)
    
    assert np.array_equal(britney_loc, test_loc_guard)
    assert np.array_equal(guard_loc, test_loc_guard)
    
    test_loc_britney = np.array([env_size-2, 1])
    test_loc_guard = np.array([env_size-3, 1])
    
    env.britney_location = test_loc_britney.copy()
    env.guard_location = test_loc_guard.copy()
    
    britney_loc, guard_loc = env.pull_britney(env.britney_location,
                                              env.guard_location)
    
    assert np.array_equal(britney_loc, np.array([7, 2]))
    assert np.array_equal(guard_loc, test_loc_britney)
    
    test_loc_britney = np.array([1, 5])
    test_loc_guard = np.array([2, 5])
    
    env.britney_location = test_loc_britney.copy()
    env.guard_location = test_loc_guard.copy()
    
    britney_loc, guard_loc = env.pull_britney(env.britney_location,
                                              env.guard_location)
    
    assert np.array_equal(test_loc_britney, guard_loc)
    assert np.array_equal(test_loc_guard, britney_loc)
    
    test_loc_britney = np.array([1, 5])
    test_loc_guard = np.array([1, 4])
    
    env.britney_location = test_loc_britney.copy()
    env.guard_location = test_loc_guard.copy()
    
    britney_loc, guard_loc = env.pull_britney(env.britney_location,
                                              env.guard_location)
    
    assert np.array_equal(britney_loc, np.array([2, 5]))
    assert np.array_equal(guard_loc, test_loc_britney)



def unittest_calculate_observations():
    observation_size = 5
    env = bw.Environment(20, 0.2, observation_size)
    env.reset()
    env.guard_location = np.array([7,7])
    observations = env.calculate_observations(observation_size)
    surroundings = observations["surroundings"]
    assert surroundings.shape == (observation_size, observation_size)
    
    observation_size = 3
    env = bw.Environment(10, 0.2, observation_size)
    env.reset()
    observations = env.calculate_observations(observation_size)
    surroundings = observations["surroundings"]
    assert surroundings.shape == (observation_size, observation_size)
    
    observation_size = 21
    env = bw.Environment(50, 0.2, observation_size)
    env.reset()
    env.guard_location = np.array([15,14])
    observations = env.calculate_observations(observation_size)
    surroundings = observations["surroundings"]
    assert surroundings.shape == (observation_size, observation_size)
    
    observation_size = 10
    env = bw.Environment(10, 0.2, observation_size)
    env.reset()
    env.guard_location = np.array([3,3])
    observations = env.calculate_observations(observation_size)
    surroundings = observations["surroundings"]
    assert surroundings.shape == (9, 9)
    
    observation_size = 21
    env = bw.Environment(50, 0.2, observation_size)
    env.reset()
    env.guard_location = np.array([1,1])
    observations = env.calculate_observations(observation_size)
    surroundings = observations["surroundings"]
    shape = math.ceil(observation_size/2)+1 # +1 because of agent's own location
    assert surroundings.shape == (shape, shape)
    
    observation_size = 21
    env = bw.Environment(50, 0.2, observation_size)
    env.reset()
    env.guard_location = np.array([48,48])
    observations = env.calculate_observations(observation_size)
    surroundings = observations["surroundings"]
    shape = math.ceil(observation_size/2)+1 # +1 because of agent's own location
    assert surroundings.shape == (shape, shape)
    
    


