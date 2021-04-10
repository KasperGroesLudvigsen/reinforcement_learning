# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:15:33 2021
@author: groes
"""

import task1.britneyworld as bw
import numpy as np
import random
import torch

class AssassinWorld(bw.Environment):
    
    def __init__(self, environment_params):
        super().__init__(environment_params)
        """ environment_params : the parameters used for britneyworld """
        self.num_assassins = 2
        self.assassin_locations = self.create_assassins(self.num_assassins)
        self.reward = 0
        self.stab_probability = 0.1
        self.stab_reward = -40
        self.dead_assassins = np.array([])
        self.dict_map_display = { 0:'.', # Nothing
                                  1:'X', # Obstacle
                                  2:'B', # Britney
                                  3:'C', # Car
                                  4:'G', # Guard
                                  5:'A'  # Assassin
                                  }
        
    def environment_step(self, guard_location, britney_location, action):
        """ guard_location and britney_location are not used in this method,
        but they are given as arguments so that in learning_environment, we
        can call environment_step() the same way regardless of the type of
        environmnt we're using (britney_world makes use of guard_location and 
        britney_location)"""
        done = False
        self.reward -= 1
        if len(self.assassin_locations) > 0:
            done, reward = self.stab(self.britney_location, self.assassin_locations,
                                     self.guard_location)
            self.reward += reward
            #self.assassin_locations = self.move_assassins(self.britney_location,
            #                                              self.assassin_locations)
        if not done:
            self.assassin_locations = self.move_assassins(self.britney_location,
                                                          self.assassin_locations)
            _, reward, done = self.take_action_guard(self.guard_location,
                                                         self.britney_location, action)
            self.reward += reward
        
        return self.reward, done
        
    def move_assassins(self, britney_location, assassin_locations):
        new_locations = []
        for loc in assassin_locations:
            if random.random() < 0.05:
                new_loc = loc + np.sign(britney_location-loc)
                new_locations.append(new_loc)
            else:
                new_locations.append(loc)
        return new_locations
                
    def stab(self, britney_location, assassin_locations, guard_location):
        guard_stabbed = False
        britney_stabbed = False
        reward = 0
        done = False
        new_assassin_locations = []
        for loc in assassin_locations:
            britney_stabbed, r = self.stab_britney(britney_location, loc)
            reward += r
            if not britney_stabbed:
                guard_stabbed, r, assassin_dead = self.stab_guard(guard_location, loc)
                if guard_stabbed:
                    done = True
                    break
                if not assassin_dead:
                    new_assassin_locations.append(loc)
                reward += r
            if guard_stabbed or britney_stabbed:
                done = True 
        self.assassin_locations = new_assassin_locations.copy()
        return done, reward
    
    def stab_britney(self, britney_location, assassin_location):
        if self.are_locations_adjacent(britney_location, assassin_location):
            done = True
            self.display()
            return done, self.stab_reward
        done = False
        reward = 0
        return done, reward
    
    def stab_guard(self, guard_location, assassin_location):
        assassin_dead = False
        if self.are_locations_adjacent(guard_location, assassin_location):
            if random.random() < self.stab_probability:
                done = True
                return done, self.stab_reward, assassin_dead
            else:
                assassin_dead = True
            self.display()
                
        done = False
        reward = 0
        return done, reward, assassin_dead
    
    def create_assassins(self, num_assassins):
        assassin_locations = []
        for i in range(num_assassins):
            location = self.get_empty_cells(1)
            # Marking location on map so that guard and Britney are not spawned
            # onto the same location. Will remove in reset
            self.map[location[0]][location[1]] = 1
            self.map[location[0]][location[1]] = 1
            assassin_locations.append(location) 
        return assassin_locations
    
    def reset(self):
        """
        This function resets the environment to a new initial state (time = 0).
        Then it places the agent and exit at new random locations.
        
        It is common practice to return the observations, 
        so that the agent can decide on the first action right after the 
        resetting of the environment.
        
        """
        self.time_elapsed = 0
        self.assassin_start_locations = self.create_assassins(self.num_assassins)
        self.assassin_locations = self.assassin_start_locations
        self.car_location = self.get_empty_cells(1)
        self.britney_start_location = self.get_empty_cells(1)
        self.britney_location = self.britney_start_location
        
        # The guard is spawned onto a location next to Britney, but cannot spawn
        # onto an obstacle
        britney_neighbors = list(self.get_neighbors(self.britney_location))
        britney_neighbors = [x for x in britney_neighbors if self.map[x[0]][x[1]] == 0]
        self.guard_start_location = np.asarray(random.choice(britney_neighbors))
        self.guard_location = self.guard_start_location
        
        # Removing assassin marks on map
        for loc in self.assassin_locations:
            self.map[loc[0]][loc[1]] = 0
        
        observations = self.calculate_observations()
        return observations
    
    def respawn(self):
        """
        Puts britney and guard and car back in original locations chosen by 
        latest reset.
        """
        self.time_elapsed = 0
        self.britney_location = self.britney_start_location
        self.guard_location = self.guard_start_location
        self.assassin_locations = self.assassin_start_locations


    def calculate_observations(self):
        relative_coordinates_car = self.car_location - self.britney_location
        relative_coordinates_britney = self.britney_location - self.guard_location
        
        relative_coordinates_assassin1 = np.array([-1,-1])
        relative_coordinates_assassin2 = np.array([-1,-1])
        
        if len(self.assassin_locations) == 2:
            relative_coordinates_assassin1 = self.britney_location - self.assassin_locations[0]
            relative_coordinates_assassin2 = self.britney_location - self.assassin_locations[1]
        if len(self.assassin_locations) == 1:
            relative_coordinates_assassin1 = self.britney_location - self.assassin_locations[0]
            
        surroundings= self.get_surroundings(
            self.guard_location, env_map=self.map, env_size=self.size, obs_size=self.obs_size)
        
        observation = {"relative_coordinates_car" : relative_coordinates_car,
                       "relative_coordinates_britney" : relative_coordinates_britney,
                       "relative_coordinates_assassin1" : relative_coordinates_assassin1,
                       "relative_coordinates_assassin2" : relative_coordinates_assassin2,
                       "surroundings" : surroundings}
        
        return observation
    
    def convert_state(self, observation, env_size = 10, norm = 2., device = "cpu"):
        # Adapted from lab 7 feedback
        """
        observation (dict) : dictionary with relative coordinates and agent's surroundings
        
        env_size (int) : length of the environment's sides
        
        norm (float) : some float, e.g. 4.0, by which to normalize surroundings
        
        device (torch object) : GPU or CPU
        
        """
        rel_coor_car = observation['relative_coordinates_car'].flatten()/env_size
        rel_coor_britney = observation['relative_coordinates_britney'].flatten()/env_size
        rel_coor_assassin1 = observation['relative_coordinates_assassin1'].flatten()/env_size
        rel_coor_assassin2 = observation['relative_coordinates_assassin2'].flatten()/env_size
        o = observation['surroundings'].flatten()/norm
        
        state_tensor = np.concatenate(
            [rel_coor_car, rel_coor_britney, rel_coor_assassin1, rel_coor_assassin2, o]
            )
        
        state_tensor = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0)        
        return state_tensor
    
    def display(self):
        # Lifted from Michael's code
        
        envir_with_agent = np.asarray(self.map).copy()
        envir_with_agent[self.guard_location[0], self.guard_location[1]] = 4
        envir_with_agent[self.britney_location[0], self.britney_location[1]] = 2
        envir_with_agent[self.car_location[0], self.car_location[1]] = 3
        for loc in self.assassin_locations:
            envir_with_agent[loc[0], loc[1]] = 5
        
        full_repr = ""

        for r in range(self.size):
            
            line = ""
            
            for c in range(self.size):

                string_repr = self.dict_map_display[ envir_with_agent[r,c] ]
                
                line += "{0:2}".format(string_repr)

            full_repr += line + "\n"

        print(full_repr)