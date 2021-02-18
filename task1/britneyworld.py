# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:13:16 2021

@author: hugha
"""
################ Rules governing movment and action of charachters ###########
'''
Guard / agent: 
    1.move to any location within a 5x5 grid of themself
    2.agent can push britney


Britney:
    1.moves randomly if not pushed by agent
    2.moves where she is pushed by agent if she is pushed.
    
NB: we should make the push variable easily upgradable to being stochastic
'''

import numpy as np
import random


class Environment:
    
    def __init__(self, N):
        
        self.map = np.zeros((N,N))
        self.size = N
        
        # creating borders
        self.map[0,:] = 1
        self.map[-1,:] = 1
        self.map[:,0] = 1
        self.map[:,-1] = 1
        
        self.guard_location = None 
        self.britney_location = None
        
        # our 'dictionary' where we assign a single number for each coordinate
        self.locations = [[x,y] for x in range(N) for y in range(N)] 
        
        self.car_location = self.get_empty_cells(1)
        
        #are training it on an environment then it should stay the same through training?
        
        # run time ###Q3: do we want this?
        self.time_elapsed = 0
        self.time_limit = self.size**2
        
        self.actions = ["up", "right", "down", "left"]
        
        self.dict_map_display = { 0:'.', # Nothing
                                  1:'X', # Obstacle
                                  2:'B', # Britney
                                  3:'C', # Car
                                  4:'G'} # Guard
                
        ### This is completely lifted from Michael's code
    def get_empty_cells(self, n_cells):
        empty_cells_coord = np.where( self.map == 0 )
        selected_indices = np.random.choice( np.arange(len(empty_cells_coord[0])), n_cells )
        selected_coordinates = empty_cells_coord[0][selected_indices], empty_cells_coord[1][selected_indices]
        
        if n_cells == 1:
            return np.asarray(selected_coordinates).reshape(2,)
        
        return selected_coordinates    

    
    # KASPER: Maybe it's better to have a move method for each agent and character
    # and make a method "get_next_position" or so that all move methods share
    def move_agent(self, action): # perhaps rename this to "agent_move" and call the "get_next_position()" instead of having the next position code inside this method
         # At every timestep, the agent receives a negative reward
        reward = -1
        bump = False
        
        next_position = self.get_next_position(action, self.guard_location)
        
        # If the agent bumps into a wall, it doesn't move, wall cells have value 1
        if self.map[next_position[0], next_position[1]] == 1:
            bump = True
        else:
            self.guard_location = next_position
        
        # calculate reward
        #britney_cell_type = self.map[self.britney_location[0], self.britney_location[1]]
        
        #if britney_cell_type == 3:
         #   reward += self.size**2
            
        
        # calculate observations
        #returns surrounding cells and relative coordinates to exit
        observations = self.calculate_observations()
        
        # update time
        self.time_elapsed += 1
        
        # verify termination condition
        done = False
        
        if self.time_elapsed == self.time_limit:
            done = True
        
        if (self.guard_location == self.car_location).all():
            done = True
            
        ###########
        
        # if BDG_loc is 1 square adjacent britney_location then britney's move is deterministic, she gets 'pushed in direction
        # if not, Britney's move is random
        
        
        
            
        return observations, reward, done    

    def get_neighbors(self, location):
        i = location[0]
        j = location[1]
        
        neighbors = {(i-1, j), (i, j+1), (i+1, j), (i, j-1)}
        
        return neighbors
        
    def are_locations_adjacent(self):
        
        neighbors = self.get_neighbors(self.britney_location)
        
        guard_location_tuple = (self.guard_location[0], self.guard_location[1])
        
        if guard_location_tuple in neighbors:
            return True
        
        return False
    
    def move_britney(self):
        if self.are_locations_adjacent():
            britney_gradient = (self.britney_location[0]-self.guard_location[0], self.britney_location[1]-self.guard_location[1])
            self.britney_location += britney_gradient
        else:
            action = random.choice(self.actions)
        
        self.britney_location = self.get_next_position(action, self.britney_location)

            
    def get_next_position(self, action, current_location):
        if action == 'up':
            return np.array( (current_location - 1, current_location[1] ) )
        if action == 'down':
            return np.array( (current_location + 1, current_location[1] ) )
        if action == 'left':
            return np.array( (current_location , current_location[1] - 1 ) )
        if action == 'right':
            return np.array( (current_location[0] , current_location[1] + 1) )
        

    def display(self):
        # Lifted from Michael's code
        
        envir_with_agent = self.map.copy()
        envir_with_agent[self.guard_location[0], self.guard_location[1]] = 4
        envir_with_agent[self.britney_location[0], self.britney_location[1]] = 2
        envir_with_agent[self.car_location[0], self.car_location[1]] = 3
        
        full_repr = ""

        for r in range(self.size):
            
            line = ""
            
            for c in range(self.size):

                string_repr = self.dict_map_display[ envir_with_agent[r,c] ]
                
                line += "{0:2}".format(string_repr)

            full_repr += line + "\n"

        print(full_repr)


    def reset(self):
        # Adapted from Michael's code
        """
        This function resets the environment to its original state (time = 0).
        Then it places the agent and exit at new random locations.
        
        It is common practice to return the observations, 
        so that the agent can decide on the first action right after the resetting of the environment.
        
        """
        self.time_elapsed = 0
        
        # position of the agent is a numpy array
        self.britney_location = np.asarray(self.get_empty_cells(1))
        
        britney_neighbors = list(self.get_neighbors(self.britney_location))
        
        self.guard_location = np.asarray(random.choice(britney_neighbors))
        
        # Calculate observations
        observations = self.calculate_observations()
        
        return observations



    def calculate_observations(self):
        
        relative_coordinates = self.car_location - self.guard_location
                
        surroundings = self.map[ self.guard_location[0] -1: self.guard_location[0] +2,
                                     self.guard_location[1] -1: self.guard_location[1] +2]
        
        obs = {'relative_coordinates':relative_coordinates,
               'surroundings': surroundings}
        
        return obs


    def run(self, action): # later on, the action should be the result of some policy
    
        reward = -1
        
        self.move_agent(action)
        self.move_britney()
        
        britney_cell_type = self.map[self.britney_location[0], self.britney_location[1]]
        
        if britney_cell_type == 3:
            reward += self.size**2

    



        
       