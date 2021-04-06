# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:15:40 2021
@author: groes
"""

import task1.assassinworld as ass
import numpy as np
import random


def get_peephole(env_size):
    return env_size//2

class AssPapWorld(ass.AssassinWorld):
    """
    Class for generating env with paparazzi. If papparazzis get to take
    a photo of Britney, the guard gets a negative reward. Paps are placed in
    the outer rim of the environment and can only takes photos through peepholes
    that are placed roughly in the middle of an axis
    """
    
    def __init__(self, environment_params, obs_size, assassin_locations,
                 stab_probability, stab_reward, num_paps, photo_reward):
        
        super().__init__(environment_params, obs_size, assassin_locations,
                         stab_probability, stab_reward)
        
        # Reward received by agent if Britney is photographed
        self.photo_reward = photo_reward
    
        # Create paparazzi
        #self.paps_generator = Paparazzi(self.size, photo_reward)
        self.paparazzi = []
        for i in range(num_paps):
            self.paparazzi.append(Paparazzi(self.size, photo_reward))
        
        # Create barriers for paparazzi - not technically necessary, but we do
        # it to be able to visualize the barriers
        self.map[2,:] = 1
        self.map[-3,:] = 1
        self.map[:,2] = 1
        self.map[:,-3] = 1
        
        # Making peepholes in the middle through which paps can take photos
        self.peephole = get_peephole(self.size)
        self.map[2, self.peephole] = 0
        self.map[-3, self.peephole] = 0
        self.map[self.peephole, 2] = 0
        self.map[self.peephole, -3] = 0
        
    def environment_step(self, action):
        done = False
        
        # Moving assassins and stabbing
        done, reward = self.stab(self.britney_location, self.assassin_locations,
                                 self.guard_location)
        self.reward += reward
        self.assassin_locations = self.move_assassins(self.britney_location,
                                                      self.assassin_locations)
        
        # Paps take photos and move
        for pap in self.paparazzi:
            photo_reward = pap.take_photo(self.guard_location, self.britney_location)
            pap.move_pap(self.map)
            self.reward += photo_reward
        
        if not done:
            _, reward, done = self.take_action_guard(self.guard_location,
                                                         self.britney_location, action)
            self.reward += reward
        
        return self.reward, done
        

class Paparazzi:
    
    """ One of these objects represents one paparazzi """
    
    def __init__(self, env_size, photo_reward):
        self.location, self.travel_direction = self.create_pap(env_size)
        self.direction_arr = self.create_directional_arr(self.travel_direction)
        self.photo_reward = photo_reward
        self.peephole = get_peephole(env_size)
        
    def create_pap(self, env_size):
        """
        Creates an initial location and a travel direction for the paparazz(o/a)
        The pap can only travel along one axis and starts by traveling in
        a randomly selected direction along that axis. The axis along which 
        the pap can travel is determined by its initial location.
        """
        
        # 1 = top row, 2 = rightmost col, 3 = buttom row, 4 = leftmost col
        self.side = random.choice([1,2,3,4]) 
        
        if self.side == 1:
            x_loc = 1
            y_loc = random.randint(3, env_size-4)
            travel_direction = random.choice(["E", "W"])
        elif self.side == 2:
            x_loc = random.randint(3, env_size-4)
            y_loc = env_size-2
            travel_direction = random.choice(["S", "N"])
        elif self.side == 3:
            x_loc = env_size-2
            y_loc = random.randint(3, env_size-4)
            travel_direction = random.choice(["E", "W"])
        else:
            x_loc = random.randint(3, env_size-4)
            y_loc = 1
            travel_direction = random.choice(["S", "N"])
        
        
        #x_loc = random.randint(0, env_size-3) # -3 because the environment's outer rim are barriers
        
        #if x_loc == 1 or x_loc == env_size-3:
        #    y_loc = random.randint(1, env_size-4)
        #    travel_direction = random.choice(["E", "W"]) # east / west
        #else:
        #    y_loc = random.choice([0, env_size-3])
        #    travel_direction = random.choice(["N", "S"]) # north / south
        
        location = np.array([x_loc, y_loc])
        
        return location, travel_direction
    
    def create_directional_arr(self, travel_direction):
        """Creating array that will move paps north, south, east or west"""
        if travel_direction == "E":
            return np.array([0, 1])
        if travel_direction == "W":
            return np.array([0, -1])
        if travel_direction == "N":
            return np.array([-1, 0])
        if travel_direction == "S":
            return np.array([1, 0])         
        
    def move_pap(self, env_map):
        """
        Method moves a pap. It creates a new location and assigns this as 
        self.location. If new location is a barrier, turn around and travel 
        the other way. Else, continue in current direction.
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

        
    def take_photo(self, guard_location, britney_location):
        """
        If pap (Paparazzi object) is by peephole and britney is in sight (i.e.
        in same col/row), the agent receives a negative reward equal to 
        photo_reward (int)
        """
        if self.pap_by_peephole() \
            and self.britney_in_sight(britney_location) \
            and not self.is_guard_blocking(guard_location, britney_location):
            return self.photo_reward
        return 0

    def britney_in_sight(self, britney_location):
        """ Checks if Britney is in the same column or row as the pap """
        if np.any(britney_location == self.peephole):
            return True
        return False


    def pap_by_peephole(self):
        if np.any(self.location == self.peephole):
            return True
        return False


    def is_guard_blocking(self, guard_location, britney_location):
        if self.side == 1 and guard_location[1] == self.peephole and guard_location[0] < britney_location[0]:
            return True
        if self.side == 2 and guard_location[0] == self.peephole and guard_location[1] > britney_location[1]:
            return True
        if self.side == 3 and guard_location[1] == self.peephole and guard_location[0] > britney_location[0]:
            return True  
        if self.side == 4 and guard_location[0] == self.peephole and guard_location[1] < britney_location[1]:
            return True
        return False