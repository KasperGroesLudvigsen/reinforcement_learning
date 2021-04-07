# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:15:33 2021
@author: groes
"""

import task1.britneyworld as bw
import numpy as np
import random
    

class AssassinWorld(bw.Environment):
    """
    assassin_locations : 2d numpy array that contains a number of (x,y) coordinates
    
    """
    def __init__(self, environment_params, obs_size, assassin_locations,
                 stab_probability, stab_reward):
        super().__init__(environment_params)

        self.assassin_locations = assassin_locations
        self.num_assassins = len(assassin_locations)
        self.reward = 0
        # Probability that agent is stabbed if adjacent to assassin
        self.stab_probability = stab_probability
        # Negative reward (int) received if agent or britney is stabbed
        self.stab_reward = stab_reward
        
    def __call__(self, action):
        done = False
        self.reward -= 1
        done, reward = self.stab(self.britney_location, self.assassin_locations)
        self.reward += reward
        self.assassin_locations = self.move_assassins(self.britney_location,
                                                      self.assassin_locations)
        if not done:
            _, reward, done = self.take_action_guard(self.guard_location,
                                                         self.britney_location, action)
            self.reward += reward
        
        return self.reward, done
        
    def move_assassins(self, britney_location, assassin_locations):
        if random.random() > 0.5:
            return assassin_locations + np.sign(britney_location-assassin_locations)
        return assassin_locations
        
    def stab(self, britney_location, assassin_locations, guard_location):
        reward = 0
        done = False
        for loc in assassin_locations:
            britney_stabbed, r = self.stab_britney(britney_location, loc)
            reward += r
            guard_stabbed, r = self.stab_guard(guard_location, loc)
            reward += r
            if guard_stabbed or britney_stabbed:
                done = True 
        return done, reward
    
    def stab_britney(self, britney_location, assassin_location):
        if self.are_locations_adjacent(britney_location, assassin_location):
            done = True
            print("Britney got stabbed")
            return done, self.stab_reward
        done = False
        reward = 0
        return done, reward
    
    def stab_guard(self, guard_location, assassin_location):
        if self.are_locations_adjacent(guard_location, assassin_location) \
            and random.random() < self.stab_probability:
                print("Guard got stabbed")
                done = True
                return done, self.stab_reward
        done = False
        reward = 0
        return done, reward