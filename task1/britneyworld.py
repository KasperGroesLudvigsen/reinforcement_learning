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
        
        self.guard_location = None #bdg = bodyguard = the agent
        self.britney_location = None
        
        # our 'dictionary' where we assign a single number for each coordinate
        self.locations = [[x,y] for x in range(N) for y in range(N)] 
        
        
        self.position_exit = self.get_empty_cells(1) ### Q2: do we have a moving exit? I think it is supposed to move but if we
        #are training it on an environment then it should stay the same through training?
        
        # run time ###Q3: do we want this?
        self.time_elapsed = 0
        self.time_limit = self.size**2
        
        ### This is completely lifted from Michael's code
    def get_empty_cells(self, n_cells):
        empty_cells_coord = np.where( self.dungeon == 0 )
        selected_indices = np.random.choice( np.arange(len(empty_cells_coord[0])), n_cells )
        selected_coordinates = empty_cells_coord[0][selected_indices], empty_cells_coord[1][selected_indices]
        
        if n_cells == 1:
            return np.asarray(selected_coordinates).reshape(2,)
        
        return selected_coordinates    

        
    def move(self, action):
         # At every timestep, the agent receives a negative reward
        reward = -1
        bump = False
        
        # action is 'up', 'down', 'left', or 'right'
        if action == 'up':
            next_position = np.array( (self.guard_location[0] - 1, self.guard_location[1] ) )
        if action == 'down':
            next_position = np.array( (self.guard_location[0] + 1, self.guard_locationguard_location[1] ) )
        if action == 'left':
            next_position = np.array( (self.guard_location[0] , self.guard_location[1] - 1 ) )
        if action == 'right':
            next_position = np.array( (self.guard_location[0] , self.guard_location[1] + 1) )
        
        # If the agent bumps into a wall, it doesn't move, wall cells have value 1
        if self.map[next_position[0], next_position[1]] == 1:
            bump = True
        else:
            self.guard_location = next_position
        
        # calculate reward
        current_cell_type = self.map[self.bdg_loc[0], self.bdg_loc[1]]
        if current_cell_type == 2:
            reward -= 20
        
        if current_cell_type == 3:
            reward += self.size**2
            
        if bump:
            reward -= 5
        
        # calculate observations
        #returns surrounding cells and relative coordinates to exit
        observations = self.calculate_observations()
        
        # update time
        self.time_elapsed += 1
        
        # verify termination condition
        done = False
        
        if self.time_elapsed == self.time_limit:
            done = True
        
        if (self.guard_location == self.position_exit).all():
            done = True
            
        ###########
        
        # if BDG_loc is 1 square adjacent britney_loc then britney's move is deterministic, she gets 'pushed in direction
        # if not, Britney's move is random
        
        
        
            
        return observations, reward, done






























        
       