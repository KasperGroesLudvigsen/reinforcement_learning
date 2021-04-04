# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:15:40 2021

@author: groes
"""

import task1.assassinworld as ass
import numpy as np
import random

class AssPapWorld(ass.AssassinWorld):
    """
    Class for generating env with paparazzi. If papparazzis get to take
    a photo of Britney, the guard gets a negative reward. Paps are placed in
    the outer rim of the environment and can only takes photos through peepholes
    that are placed roughly in the middle of an axis
    """
    
    def __init__(self, N, stumble_prob, assassin_locations, num_paps, photo_reward):
        super().__init__(N, stumble_prob, assassin_locations)
        
        # Reward received by agent if Britney is photographed
        self.photo_reward = photo_reward
    
        # Create paparazzi
        self.paps_generator = Paparazzi()
        self.paparazzi = []
        for i in range(num_paps):
            self.paparazzi.append(self.paps_generator())
        
        # Create barriers for paparazzi - not technically necessary, but we do
        # it to be able to visualize the barriers
        self.map[2,:] = 1
        self.map[-3,:] = 1
        self.map[:,2] = 1
        self.map[:,-3] = 1
        
        # Making peepholes in the middle through which paps can take photos
        self.peephole = self.size//2
        self.map[2, self.peephole] = 0
        self.map[-3, self.peephole] = 0
        self.map[self.peephole, 2] = 0
        self.map[self.peephole, -3] = 0
        

                

class Paparazzi:
    
    """ One of these objects represents one paparazzi """
    
    def __init__(self, env_size, photo_reward):
        self.location, self.travel_direction = self.create_pap(env_size)
        self.direction_arr = self.create_directional_arr(self.travel_direction)
        self.photo_reward = photo_reward
        
    def create_pap(self, env_size):
        """
        Creates an initial location and a travel direction for the paparazz(o/a)
        The pap can only travel along one axis and starts by traveling in
        a randomly selected direction along that axis. The axis along which 
        the pap can travel is determined by its initial location.
        """
        x_loc = random.randint(0, env_size-3) # -3 because the environment's outer rim are barriers
        
        if x_loc == 0 or x_loc == env_size-3:
            y_loc = random.randint(1, env_size-4)
            travel_direction = random.choice(["E", "W"]) # east / west
        else:
            y_loc = random.choice([0, env_size-3])
            travel_direction = random.choice(["N", "S"]) # north / south
        
        location = np.array([x_loc, y_loc])
        
        return location, travel_direction
    
    def create_directional_arr(self, travel_direction):
        """Creating array that will move paps north, south, east or west"""
        if travel_direction == "E":
            return np.array([0, 1])
        if travel_direction == "W":
            return np.array([0, -1])
        if travel_direction == "N":
            return np.array([1, 0])
        if travel_direction == "S":
            return np.array([-1, 0])         
        
    def move_pap(self, env_map):
        """
        Method moves a pap. It creates a new location. If new location is a
        barrier, turn around and travel the other way. Else, continue in current
        direction.
        """
        new_location = self.location + self.direction_arr
        if env_map[new_location[0], new_location[1]] == 1:
            self.travel_direction = self.change_direction(self.travel_direction)
            self.direction_arr = self.create_directional_arr(self.travel_direction)
            self.location += self.direction_arr
        else:
            self.location = new_location
         
    def change_direction(self, current_direction):
        if current_direction == "E":
            return "W"
        if current_direction == "W":
            return "E"
        if current_direction == "N":
            return "S"
        if current_direction == "S":
            return "N"

        
    def take_photo(self, pap, photo_reward):
        """
        If pap (Paparazzi object) is by peephole and britney is in sight (i.e.
        in same col/row), the agent receives a negative reward equal to 
        photo_reward (int)
        """
        
        if np.any(pap.location == self.peephole):
            britney_got_papped = self.britney_in_sight(pap.location)#and np.any(self.britney_location == self.peephole):
            if britney_got_papped:
                return self.photo_reward
        else:
            return 0

    def britney_in_sight(self, britney_location, pap_location):
        if britney_location[0] == pap_location[0] \
            or britney_location[1] == pap_location[1]:
                return True
        else:
            return False






    