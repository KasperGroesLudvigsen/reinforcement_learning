# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 16:33:18 2021

@author: groes
"""
import task1.assassinworld as assworld
import task1.asspapsworld as asspap
import numpy as np
from copy import copy
import math 

env_params = {
    'N' : 20,
    'stumble_prob' : 0.3
    }

ass_locations = [np.array([5,5]), np.array([6,6])]

num_paps = 4
photo_reward = -5
stab_probability = 0.1
stab_reward = -20

env = asspap.AssPapWorld(
    environment_params=env_params, obs_size=5,assassin_locations=ass_locations,
    stab_probability=stab_probability, stab_reward=stab_reward, 
    num_paps=num_paps, photo_reward=photo_reward
    )


def test_asspap_attributes():
    assert len(env.paparazzi) == num_paps
    assert np.array_equal(env.assassin_locations[0], ass_locations[0])
    assert np.array_equal(env.assassin_locations[1], ass_locations[1])

test_asspap_attributes()

def test_create_paps():
    for i in range(100):
        pap = asspap.Paparazzi(env_params["N"], photo_reward)
        if pap.location[0] != 1 and pap.location[0] != env_params["N"]-2:
            assert pap.location[1] == 1 or pap.location[1] == env_params["N"]-2
  
test_create_paps()

def test_move_pap():
    
    # Asserting that moving east works as expected
    pap = asspap.Paparazzi(env_params["N"], photo_reward)
    pap.travel_direction = "E"
    pap.location = np.array([1, 5])
    pap.direction_arr = pap.create_directional_arr(pap.travel_direction)
    pap.move_pap(env.map)
    assert np.array_equal(pap.location, np.array([1, 6]))
    
    # Asserting that moving west works as expected
    pap = asspap.Paparazzi(env_params["N"], photo_reward)
    pap.travel_direction = "W"
    pap.location = np.array([1, 5])
    pap.direction_arr = pap.create_directional_arr(pap.travel_direction)
    pap.move_pap(env.map)
    assert np.array_equal(pap.location, np.array([1, 4]))
    
    # Asserting that moving north works as expected
    pap = asspap.Paparazzi(env_params["N"], photo_reward)
    pap.travel_direction = "N"
    pap.location = np.array([7, 1])
    pap.direction_arr = pap.create_directional_arr(pap.travel_direction)
    pap.move_pap(env.map)
    assert np.array_equal(pap.location, np.array([6, 1]))
    
    # Asserting that moving south works as expected
    pap = asspap.Paparazzi(env_params["N"], photo_reward)
    pap.travel_direction = "S"
    pap.location = np.array([7, 1])
    pap.direction_arr = pap.create_directional_arr(pap.travel_direction)
    pap.move_pap(env.map)
    assert np.array_equal(pap.location, np.array([8, 1]))
    
    # Asserting that pap changes direction if new location is obstacle
    pap = asspap.Paparazzi(env_params["N"], photo_reward)
    pap.travel_direction = "W"
    pap.location = np.array([1, 3])
    pap.direction_arr = pap.create_directional_arr(pap.travel_direction)
    pap.move_pap(env.map)
    assert pap.travel_direction == "E"
    assert np.array_equal(pap.location, np.array([1, 4]))
    
    # Asserting that pap changes direction if new location is obstacle
    pap = asspap.Paparazzi(env_params["N"], photo_reward)
    pap.travel_direction = "N"
    pap.location = np.array([3, 1])
    pap.direction_arr = pap.create_directional_arr(pap.travel_direction)
    pap.move_pap(env.map)
    assert pap.travel_direction == "S"
    assert np.array_equal(pap.location, np.array([4, 1]))
    
    # Asserting that paps always move
    previous_location = copy(pap.location)
    for i in range(5000):
        pap.move_pap(env.map)
        assert not np.array_equal(previous_location, pap.location)
        previous_location = copy(pap.location)

test_move_pap() 
    

def test_take_photo():
    # Asserting that reward is negative if Britney is in line of sight and pap
    # is by peephole
    guard_location = np.array([2,2])
    
    pap = asspap.Paparazzi(env_params["N"], photo_reward)
    britney_location = np.array([6, pap.peephole])
    pap.location = np.array([1, pap.peephole])
    reward = pap.take_photo(guard_location, britney_location)
    assert reward == photo_reward
    
    # Asserting that reward is zero if Britney is not in line of sight
    #peephole = env.peephole # 10
    pap = asspap.Paparazzi(env_params["N"], photo_reward)
    britney_location = np.array([6, 2])
    pap.location = np.array([1, pap.peephole])
    reward = pap.take_photo(guard_location, britney_location)
    assert reward == 0
    
    # Asserting that reward is zero if Britney is in same col as peephole,
    # but pap is not by peephole
    #peephole = env.peephole # 10
    pap = asspap.Paparazzi(env_params["N"], photo_reward)
    britney_location = np.array([6, pap.peephole])
    pap.location = np.array([1, 9])
    reward = pap.take_photo(guard_location, britney_location)
    assert reward == 0
  
test_take_photo()

def test_britney_in_sight():
    #peephole = env.peephole # 10
    pap = asspap.Paparazzi(env_params["N"], photo_reward)
    britney_location = np.array([6, pap.peephole])
    in_sight = pap.britney_in_sight(britney_location)
    assert in_sight
    
    britney_location = np.array([pap.peephole, 4])
    in_sight = pap.britney_in_sight(britney_location)
    assert in_sight
    
    britney_location = np.array([pap.peephole-1, 4])
    in_sight = pap.britney_in_sight(britney_location)
    assert not in_sight
    
    britney_location = np.array([pap.peephole+1, 14])
    in_sight = pap.britney_in_sight(britney_location)
    assert not in_sight

test_britney_in_sight()
    
def test_pap_by_peephole():
    pap = asspap.Paparazzi(env_params["N"], photo_reward)
    peephole = pap.peephole
    pap.location = np.array([peephole, 4])
    assert pap.pap_by_peephole()
    
    pap.location = np.array([7, peephole])
    assert pap.pap_by_peephole()
    
    pap.location = np.array([9,9])
    assert not pap.pap_by_peephole()
    
    pap.location = np.array([2,17])
    assert not pap.pap_by_peephole()
    
test_pap_by_peephole()   
  
def test_env_step_moves_paps():
    """ Testing that env.environment_step() moves paps """
    
    env = asspap.AssPapWorld(environment_params=env_params, obs_size=5, 
                         assassin_locations=ass_locations,
                         stab_probability=stab_probability,
                         stab_reward=stab_reward,
                         num_paps=4, photo_reward=photo_reward)
    
    pap1_prev_loc = copy(env.paparazzi[0].location)
    pap2_prev_loc = copy(env.paparazzi[1].location)
    pap3_prev_loc = copy(env.paparazzi[2].location)
    pap4_prev_loc = copy(env.paparazzi[3].location)
    
    for _ in range(1000):
        env.environment_step("S") # agent takes some arbitrary action
        assert not np.array_equal(pap1_prev_loc, env.paparazzi[0].location)
        assert not np.array_equal(pap2_prev_loc, env.paparazzi[1].location) 
        assert not np.array_equal(pap3_prev_loc, env.paparazzi[2].location)
        assert not np.array_equal(pap4_prev_loc, env.paparazzi[3].location)
        pap1_prev_loc = copy(env.paparazzi[0].location)
        pap2_prev_loc = copy(env.paparazzi[1].location)
        pap3_prev_loc = copy(env.paparazzi[2].location)
        pap4_prev_loc = copy(env.paparazzi[3].location)

test_env_step_moves_paps()        
  
def test_asspapworld_step():
    env = asspap.AssPapWorld(environment_params=env_params, obs_size=5, 
                             assassin_locations=ass_locations,
                             stab_probability=stab_probability,
                             stab_reward=stab_reward,
                             num_paps=4, photo_reward=photo_reward)
    
    ## Testing that actors move 
    initial_britney_loc = np.array([11,12])
    initial_guard_loc = np.array([11,11])
    env.guard_location = initial_guard_loc
    env.britney_location = initial_britney_loc
    env.environment_step("S")
    assert np.array_equal(env.guard_location, np.array([12,11]))
    
    ## Testing that agent does not move onto Britney
    env.britney_location = np.array([13,11])
    env.environment_step("S")
    assert np.array_equal(env.guard_location, np.array([12,11]))
    
    ## Testing that the right reward is returned
    true_reward = 0
    env.reward = 0
    # Subtracting 5 because britney will get photographed once
    env.britney_location = np.array([env.peephole, 10])
    env.paparazzi[0].location = np.array([env.peephole, 1])
    env.paparazzi[1].location = np.array([env.peephole+1, 1])
    env.paparazzi[2].location = np.array([env.peephole+2, 1])
    env.paparazzi[3].location = np.array([env.peephole-1, 1])
    env.car_location = np.array([3,3])
    true_reward += photo_reward
    # Subtracting 1 because a negative reward is received for each time step
    env.environment_step("S")
    true_reward -= 1
    assert env.reward == true_reward
    
    ## Testing that the right reward is returned
    true_reward = 0
    env.reward = 0
    # Subtracting 5 because britney will get photographed once
    env.britney_location = np.array([env.peephole, 10])
    env.assassin_locations = [np.array([2,2]), np.array([3,3])]
    env.paparazzi[0].location = np.array([env.peephole, 1])
    env.paparazzi[1].location = np.array([env.peephole, env.size-2])
    env.paparazzi[2].location = np.array([env.peephole+2, 1])
    env.paparazzi[3].location = np.array([env.peephole-1, 1])
    env.car_location = np.array([3,3])
    true_reward += photo_reward*2
    # Subtracting 1 because a negative reward is received for each time step
    env.environment_step("S")
    true_reward -= 1
    assert env.reward == true_reward
    
    ## Testing that the right reward is returned
    true_reward = 0
    env.reward = 0
    # Subtracting 2*-5 because britney will get photographed twice
    env.guard_location = np.array([17,17])
    env.assassin_locations = [np.array([2,2]), np.array([3,3])]
    env.britney_location = np.array([env.peephole, 10])
    env.paparazzi[0].location = np.array([env.peephole, 1])
    env.paparazzi[1].location = np.array([1, env.peephole])
    env.paparazzi[2].location = np.array([env.peephole+4, 1])
    env.paparazzi[3].location = np.array([env.peephole-4, 1])
    env.car_location = np.array([3,3])
    true_reward += (env.photo_reward*2)
    # Subtracting 1 because a negative reward is received for each time step
    env.environment_step("S")
    true_reward -= 1
    assert env.reward == true_reward
    
test_asspapworld_step()

def test_ass_stab():
    steps = 1000
    times_britney_stabbed = 0
    times_guard_stabbed = 0
    env = assworld.AssassinWorld(environment_params=env_params, obs_size=5,
                                 assassin_locations=ass_locations, 
                                 stab_probability=stab_probability,
                                 stab_reward=stab_reward)
    
    # Asserting that Britney is stabbed every time she's adjacent to assassin
    # and that reward is -40,000
    total_reward = 0
    guard_location = np.array([10,10])
    britney_location = ass_locations[0] + np.array([1, 0])
    for i in range(steps):
        done, reward = env.stab(britney_location, ass_locations, guard_location)
        total_reward += reward
        if done:
            times_britney_stabbed += 1
    assert times_britney_stabbed == steps
    assert total_reward == steps*stab_reward*2 # She's adjacent to both assassins
    
    # Asserting that Britney is stabbed every time she's adjacent to assassin
    # and that reward is -20,000
    total_reward = 0
    times_britney_stabbed = 0

    guard_location = np.array([10,10])
    britney_location = ass_locations[0] - np.array([0, 1])
    for i in range(steps):
        done, reward = env.stab(britney_location, ass_locations, guard_location)
        total_reward += reward
        if done:
            times_britney_stabbed += 1
    assert times_britney_stabbed == steps
    assert total_reward == steps*stab_reward # She's only adjacent to one assassin
    
    # Asserting that Britney does not get stabbed if she's not adjacent to assassin
    guard_location = np.array([10,10])
    britney_location = np.array([2,2])
    times_britney_stabbed = 0
    for i in range(steps):
        done, reward = env.stab(britney_location, ass_locations, guard_location)
        assert reward == 0
        if done:
            times_britney_stabbed += 1
    assert times_britney_stabbed == 0
    
    # Asserting that guard is stabbed at least once
    total_reward = 0
    britney_location = np.array([2,2])
    guard_location = ass_locations[0] + np.array([1, 0])
    for i in range(steps):    
        done, reward = env.stab(britney_location, ass_locations, guard_location)
        total_reward += reward
        if done:
            times_guard_stabbed += 1
    assert times_guard_stabbed > 0
    assert total_reward < 0
         
test_ass_stab()
        
def test_agent_blocks_photo():
    pap = asspap.Paparazzi(env_size=env_params["N"], photo_reward=photo_reward)
    
    # Asserting that photo is blocked when guard and britney are in peephole
    # column and guard is closer to peephole than Britney
    pap.side = 1 # top row
    guard_location = np.array([4, pap.peephole])
    britney_location = np.array([5, pap.peephole])
    blocked = pap.is_guard_blocking(guard_location, britney_location)
    assert blocked
    guard_location = np.array([4, pap.peephole])
    britney_location = np.array([11, pap.peephole])
    blocked = pap.is_guard_blocking(guard_location, britney_location)
    assert blocked
    # Asserting that photo is not blocked if britney is in peephole
    # column and but guard is not
    guard_location = np.array([4, pap.peephole+1])
    britney_location = np.array([11, pap.peephole])
    blocked = pap.is_guard_blocking(guard_location, britney_location)
    assert not blocked
    
    # Repeat above pattern for each of the sides that the pap can be in
    pap.side = 2 # right-most col
    guard_location = np.array([pap.peephole, 5])
    britney_location = np.array([pap.peephole, 4])
    blocked = pap.is_guard_blocking(guard_location, britney_location)
    assert blocked
    guard_location = np.array([pap.peephole-1, 5])
    britney_location = np.array([pap.peephole, 4])
    blocked = pap.is_guard_blocking(guard_location, britney_location)
    assert not blocked
        
    pap.side = 3 # bottom row
    guard_location = np.array([5, pap.peephole])
    britney_location = np.array([4, pap.peephole])
    blocked = pap.is_guard_blocking(guard_location, britney_location)
    assert blocked
    guard_location = np.array([pap.peephole-1, 5])
    britney_location = np.array([pap.peephole, 4])
    blocked = pap.is_guard_blocking(guard_location, britney_location)
    assert not blocked
    
    pap.side = 4 # left-most c
    guard_location = np.array([pap.peephole, 4])
    britney_location = np.array([pap.peephole, 5])
    blocked = pap.is_guard_blocking(guard_location, britney_location)
    assert blocked
    guard_location = np.array([pap.peephole-1, 5])
    britney_location = np.array([pap.peephole, 4])
    blocked = pap.is_guard_blocking(guard_location, britney_location)
    assert not blocked

test_agent_blocks_photo()    


env.map

        
def get_surroundings(guard_location, env_map, env_size, obs_size):
    lower_bound = math.floor(obs_size / 2)
    upper_bound = lower_bound + 1 # +1 because slicing np.arrays is not inclusive
    
    row_start, row_end = guard_location[0] -lower_bound, guard_location[0] +upper_bound
    col_start, col_end = guard_location[1] -lower_bound, guard_location[1] +upper_bound
    
    if row_start < 0:
        diff = abs(row_start)
        row_start = 0
        row_end += diff
    if row_end > env_size:
        diff = row_end-env_size
        row_end = env_size
        row_start -= diff
    if col_start < 0:
        diff = abs(col_start)
        col_start = 0
        col_end += diff
    if col_end > env_size:
        diff = col_end-env_size
        col_end = env_size
        col_start -= diff
        
    surroundings = env_map[row_start:row_end, col_start:col_end]
    
    return surroundings


def test_get_surroundings():
    guard_location = np.array([2,2])
    obs_size = env.size//2 # the agent sees 1/4 of the env
    
    surroundings = get_surroundings(guard_location=guard_location, 
                                    env_map=env.map,
                                    env_size=env.size,
                                    obs_size=obs_size)
    assert surroundings.shape == (11, 11)
    # Asserting that all values in first tensor is 1s because the first
    # tensor represents the border of the env
    for i in surroundings[0]:
        assert i == 1
        
        
    guard_location = np.array([17,17])
    obs_size = env.size//2 # the agent sees 1/4 of the env
    
    surroundings = get_surroundings(guard_location=guard_location, 
                                    env_map=env.map,
                                    env_size=env.size,
                                    obs_size=obs_size)
    
    assert surroundings.shape == (11, 11)
    # Asserting that all values in first tensor is 1s because the first
    # tensor represents the border of the env
    for i in surroundings[-1]:
        assert i == 1

    guard_location = np.array([10,10])
    obs_size = env.size//2 # the agent sees 1/4 of the env
    
    surroundings = get_surroundings(guard_location=guard_location, 
                                    env_map=env.map,
                                    env_size=env.size,
                                    obs_size=obs_size)
    
    assert surroundings.shape == (11, 11)
    # Asserting that all values in first tensor is 0s
    for i in surroundings[0]:
        assert i == 0        
    
    
    
    




                      
    