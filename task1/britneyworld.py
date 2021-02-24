# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:13:16 2021

@author: hugha
"""
################ Rules governing movment and action of charachters ###########


"""
See movement mechanism specifications in report
"""

import numpy as np
import random


class Environment:
    
    def __init__(self, N):
        
        if N < 4:
            raise Exception('N must be larger than 3')
        
        self.map = np.zeros((N,N))
        self.size = N
        
        # creating borders
        self.map[0,:] = 1
        self.map[-1,:] = 1
        self.map[:,0] = 1
        self.map[:,-1] = 1
        
        self.guard_location = None 
        self.britney_location = None
        self.car_location = None
        self.reward = None
        # our 'dictionary' where we assign a single number for each coordinate
        self.locations = [[x,y] for x in range(N) for y in range(N)] 
        
        
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
                
        
    def get_empty_cells(self, n_cells):
        ### This is completely lifted from Michael's code
        empty_cells_coord = np.where( self.map == 0 )
        selected_indices = np.random.choice( np.arange(len(empty_cells_coord[0])), n_cells )
        selected_coordinates = empty_cells_coord[0][selected_indices], empty_cells_coord[1][selected_indices]
        
        if n_cells == 1:
            return np.asarray(selected_coordinates).reshape(2,)
        
        return selected_coordinates    

    

    def move_agent(self, action):
         # At every timestep, the agent receives a negative reward
        reward = -1
        #bump = False
        
        next_position = self.get_next_position(action, self.guard_location)
        
        # agent only moves into next position if it is open space
        if self.map[next_position[0], next_position[1]] == 0:
            self.guard_location = next_position
        
        # calculate observations
        # returns surrounding cells and relative coordinates to exit
        observations = self.calculate_observations()
        
        # update time
        self.time_elapsed += 1
        
        # verify termination condition
        done = False
        
        if self.time_elapsed == self.time_limit:
            print("Time limit reached")
            done = True
        
        # changed from agent_location to britney_location as the objective 
        # is to get britney to the car
        if (self.britney_location == self.car_location).all():
            reward += self.size**2
            print("Britney got to her car safely")
            done = True
            
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
            new_britney_location = self.britney_location + britney_gradient

        else:
            action = random.choice(self.actions)
            new_britney_location = self.get_next_position(action, self.britney_location)
        
        if self.map[new_britney_location[0]][new_britney_location[1]] != 1:
            self.britney_location = new_britney_location

            
    def get_next_position(self, action, current_location):
        if action == 'N': # North, i.e. up
            return np.array((current_location[0] - 1, current_location[1]))
        if action == 'S': # South, i.e. down
            return np.array((current_location[0] + 1, current_location[1] ))
        if action == 'W': # West, i.e. left
            return np.array((current_location[0] , current_location[1] - 1 ))
        if action == 'E': # East, i.e. right
            return np.array((current_location[0] , current_location[1] + 1))
        if action == 'NE': # North east
            return np.array((current_location[0] - 1, current_location[1] + 1))
        if action == "SE": # South east
            return np.array((current_location[0] + 1, current_location[1] + 1))
        if action == "SW": # South west
            return np.array((current_location[0] + 1, current_location[1] - 1))
        if action == "NW": # North west
            return np.array((current_location[0] - 1, current_location[1] -1))
        
        
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
        self.britney_location = self.get_empty_cells(1)

        
        # The guard is spawned onto a location next to Britney, but cannot spawn
        # onto an obstacle
        britney_neighbors = list(self.get_neighbors(self.britney_location))
        britney_neighbors = [x for x in britney_neighbors if self.map[x[0]][x[1]] == 0]
        self.guard_location = np.asarray(random.choice(britney_neighbors))

        # Setting car location 
        self.car_location = self.get_empty_cells(1)
        
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

    # We should consider whether we actually need this method. Not much happens in
    # that could couldnt be handled outside the class methods
    def run(self, action): 
        
        done = False

        observations, reward, done = self.move_agent(action)
        self.move_britney()
        
        #if (self.britney_location == self.car_location).all():
        #    reward += self.size**2
        #    done = True
        
        return reward, done
    
    



        
       