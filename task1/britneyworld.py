# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:13:16 2021

@author: hugha
"""

"""
See movement mechanism specifications in report
"""

import numpy as np
import random


class Environment:
    
    
    
    def __init__(self, N, stumble_prob):
        
        if N < 4:
            raise Exception('N must be larger than 3')
        
        self.map = np.zeros((N,N))
        self.size = N
        
        # creating borders
        self.map[0,:] = 1
        self.map[-1,:] = 1
        self.map[:,0] = 1
        self.map[:,-1] = 1
        
        self.britney_start_location = np.array([2,2])
        self.guard_start_location = np.array([1,1])
        
        self.guard_location = self.britney_start_location 
        self.britney_location = self.guard_start_location
        self.stumple_prob = stumble_prob
        self.car_location = np.array([N-2, N-2])
        self.reward = None
        # our 'dictionary' where we assign a single number for each coordinate
        #self.locations = [[x,y] for x in range(N) for y in range(N)] 
        index_states = np.arange(0, (N*N)**2)
        np.random.shuffle(index_states)
        self.states = index_states.reshape(N,N,N,N)
        #are training it on an environment then it should stay the same through training?
        
        # run time ###Q3: do we want this?
        self.time_elapsed = 0
        self.guard_actions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "push"]
        self.time_limit = self.size**2
        self.britney_actions = ["N", "S", "E", "W"]
        self.dict_map_display = { 0:'.', # Nothing
                                  1:'X', # Obstacle
                                  2:'B', # Britney
                                  3:'C', # Car
                                  4:'G'} # Guard
                
    def push_britney(self, britney_location, guard_location):
        britney_gradient = (britney_location[0]-guard_location[0], britney_location[1]-guard_location[1]) 
        britney_new_location = britney_location + britney_gradient
        
        if self.is_empty(britney_new_location): # she can fall onto agent
                return britney_new_location
        return britney_location
        
    def get_empty_cells(self, n_cells):
        ### This is completely lifted from Michael's code
        empty_cells_coord = np.where( self.map == 0 )
        selected_indices = np.random.choice( np.arange(len(empty_cells_coord[0])), n_cells )
        selected_coordinates = empty_cells_coord[0][selected_indices], empty_cells_coord[1][selected_indices]
        
        if n_cells == 1:
            return np.asarray(selected_coordinates).reshape(2,)
        
        return selected_coordinates    
            
        
    def take_action_guard(self, guard_location, britney_location, action):
        # At every timestep, the agent receives a negative reward
        reward = -1
        self.time_elapsed += 1        
        
        if action == "push":
            if self.are_locations_adjacent(britney_location, guard_location):#(self.britney_location, self.guard_location):
                britney_location = self.push_britney(britney_location, guard_location)
        else:
            # agent only moves into next position if it is open space
            #if self.is_empty(next_position):
            guard_location = self.get_next_position(action, guard_location)

        britney_location = self.britney_stumbles(self.stumple_prob, britney_location)
    
        done = False
    
        if self.time_elapsed == self.time_limit:
            #print("Time limit reached")
            done = True
        
        if (britney_location == self.car_location).all():
            reward += self.size**2
            #print("Britney got to her car safely")
            done = True
        
        self.britney_location = britney_location
        self.guard_location = guard_location
        
        state = self.states[guard_location[0]][guard_location[1]][britney_location[0]][britney_location[1]]
        
        return state, reward, done    

    def is_empty(self, position):
        """
        Helper function for get_next_position() to check if a call is empty, 
        i.e. no other agent is on it
        """
        a = self.map[position[0]][position[1]]==0
        b = position[0] == self.guard_location[0] and position[1] == self.guard_location[1] 
        c = position[0] == self.britney_location[0] and position[1] == self.britney_location[1] 
        return a and not b and not c
        
        
    def get_neighbors(self, location):
        i = location[0]
        j = location[1]
        
        neighbors = {(i-1, j), (i, j+1), (i+1, j), (i, j-1), (i-1, j+1), 
                     (i+1, j+1), (i+1, j-1), (i-1, j-1)}
        
        return neighbors
        
    def are_locations_adjacent(self, location1, location2):
        neighbors = self.get_neighbors(location1)
        location2_tuple = (location2[0], location2[1])
        
        if location2_tuple in neighbors:
            return True
        
        return False
    
    def britney_stumbles(self, stumble_probability, britney_location):
        """ For every t, there is a probability that Britney stumbles to a new location """
        if stumble_probability >= random.random():
            action = random.choice(self.britney_actions)
            return self.get_next_position(action, britney_location)
        return britney_location
        
            
    def get_next_position(self, action, current_location):
        """
        Computes next location given some action and returns that location if
        it is empty. If it is not empty, e.g. if britney is on it, return 
        current location. 
        """
        if action == 'N': # North, i.e. up
            next_location = np.array((current_location[0] - 1, current_location[1]))
            if self.is_empty(next_location):
                return next_location
        if action == 'S': # South, i.e. down
            next_location = np.array((current_location[0] + 1, current_location[1] ))
            if self.is_empty(next_location):
                return next_location
        if action == 'W': # West, i.e. left
            next_location = np.array((current_location[0] , current_location[1] - 1 ))
            if self.is_empty(next_location):
                return next_location
        if action == 'E': # East, i.e. right
            next_location = np.array((current_location[0] , current_location[1] + 1))
            if self.is_empty(next_location):
                return next_location
        if action == 'NE': # North east
            next_location = np.array((current_location[0] - 1, current_location[1] + 1))
            if self.is_empty(next_location):
                return next_location
        if action == "SE": # South east
            next_location = np.array((current_location[0] + 1, current_location[1] + 1))
            if self.is_empty(next_location):
                return next_location
        if action == "SW": # South west
            next_location = np.array((current_location[0] + 1, current_location[1] - 1))
            if self.is_empty(next_location):
                return next_location
        if action == "NW": # North west
            next_location = np.array((current_location[0] - 1, current_location[1] -1))
            if self.is_empty(next_location):
                return next_location        

        return current_location
        
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
        so that the agent can decide on the first action right after the 
        resetting of the environment.
        
        """
        self.time_elapsed = 0
        
        # Setting Britney's location
        self.britney_start_location = self.get_empty_cells(1)
        #self.britney_start_location = np.array([2,2])
        self.britney_location = self.britney_start_location
        
        # The guard is spawned onto a location next to Britney, but cannot spawn
        # onto an obstacle
        britney_neighbors = list(self.get_neighbors(self.britney_location))
        britney_neighbors = [x for x in britney_neighbors if self.map[x[0]][x[1]] == 0]
        self.guard_start_location = np.asarray(random.choice(britney_neighbors))
        self.guard_location = self.guard_start_location
        # Setting car location 
        self.car_location = self.get_empty_cells(1)
        #self.car_location = np.array([self.size-3,self.size-3])
        
        # Calculate observations
        observations = self.calculate_observations()
        
        return observations
    
    def respawn(self):
        """
        Puts britney and guard back in original locations
        """
        self.time_elapsed = 0
        self.britney_location = self.britney_start_location
        self.guard_location = self.guard_start_location
        

    def calculate_observations(self):
        
        relative_coordinates = self.car_location - self.guard_location
        surroundings = self.map[ self.guard_location[0] -1: self.guard_location[0] +2,
                                     self.guard_location[1] -1: self.guard_location[1] +2]
        
        obs = {'relative_coordinates':relative_coordinates,
               'surroundings': surroundings}
        
        return obs
    
    def get_state(self):
        state = self.states[self.guard_location[0]][self.guard_location[1]]\
            [self.britney_location[0]][self.britney_location[1]]
        return state
    


       