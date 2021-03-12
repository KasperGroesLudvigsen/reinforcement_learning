# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:15:33 2021

@author: groes
"""

import task1.britneyworld as bw
import numpy as np

    


class AssassinWorld(bw.Environment):
    """
    assassin_locations : 2d numpy array that contains a number of (x,y) coordinates
    
    """
    def __init__(self, N, stumble_prob, assassin_locations):
        super().__init__(N, stumble_prob)

        self.assassin_locations = assassin_locations
        self.num_assassins = len(assassin_locations)
        self.reward = 0
        
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
        return assassin_locations + np.sign(britney_location-assassin_locations)
        
    def stab(self, britney_location, assassin_locations):
        for loc in assassin_locations:
            if self.are_locations_adjacent(britney_location, loc):
                done = True
                reward = -20
            else:
                done = False
                reward = 0
                
        return done, reward
            


    
        
