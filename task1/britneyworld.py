# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:13:16 2021

@author: hugha
"""
import numpy as np
import random
import math
import torch

class Environment:
    
    def __init__(self, environment_params):
        
        if environment_params['N'] < 4:
            raise Exception('N must be larger than 3')
        
        self.map = torch.zeros((environment_params['N'],environment_params['N']))
        self.size = environment_params['N']
        self.obs_size = environment_params['observation_size']
        
        # creating borders
        self.map[0,:] = 1
        self.map[-1,:] = 1
        self.map[:,0] = 1
        self.map[:,-1] = 1
        
        self.britney_start_location = None
        self.guard_start_location = None
        
        self.guard_location = self.britney_start_location 
        self.britney_location = self.guard_start_location
        
        # This is the probability that britney makes a random move
        self.stumple_prob = environment_params['stumble_prob']
        
        self.car_location = None
        self.reward = None
        
        self.time_elapsed = 0
        self.guard_actions =["N","NE","E","SE","S","SW","W","NW","push","pull"]
        self.time_limit = self.size**2
        self.britney_actions = ["N", "S", "E", "W"]
        self.dict_map_display = { 0:'.', # Nothing
                                  1:'X', # Obstacle
                                  2:'B', # Britney
                                  3:'C', # Car
                                  4:'G'} # Guard

        # For task2
        index_states = np.arange(
            0, (environment_params['N']*environment_params['N'])**2
            )
        np.random.shuffle(index_states)
        self.states = index_states.reshape(
            environment_params['N'],environment_params['N'],\
                environment_params['N'],environment_params['N']
                )
                
        
    def push_britney(self, britney_location, guard_location, car_location):
        closeness_before = sum((britney_location-car_location)**2)
        reward = 0
        britney_gradient = (
            britney_location[0]-guard_location[0], \
                britney_location[1]-guard_location[1]
                ) 
        britney_new_location = britney_location + britney_gradient 
        closeness_after = sum((britney_new_location-car_location)**2)
        if closeness_after < closeness_before:
            reward = 1
        if self.is_empty(britney_new_location): # she can fall onto agent
                return britney_new_location, reward
        return britney_location, reward
        
    def get_empty_cells(self, n_cells):
        # This is completely lifted from Michael's code
        empty_cells_coord = np.where( self.map == 0 )
        selected_indices = np.random.choice(
            np.arange(len(empty_cells_coord[0])), n_cells
            )
        selected_coordinates = empty_cells_coord[0][selected_indices], \
            empty_cells_coord[1][selected_indices]
        if n_cells == 1:
            return np.asarray(selected_coordinates).reshape(2,)
        return selected_coordinates   
    
    def environment_step(self, guard_location, britney_location, action):
        '''
        This is so that we can call AssassinWorld, AssaPapWorld and 
        BritneyWorld with the same method.
        '''
        _, reward, done = self.take_action_guard(
            guard_location, britney_location, action
            ) 
        return reward, done
        
    def take_action_guard(self, guard_location, britney_location, action):
        # At every timestep, the agent receives a negative reward
        reward = -1.0
        self.time_elapsed += 1        
        
        if action == "push":
            if self.are_locations_adjacent(britney_location, guard_location):
                britney_location, push_reward = self.push_britney(
                    britney_location, guard_location
                    )
                reward += push_reward
        elif action == "pull":
            if self.are_locations_adjacent(britney_location, guard_location):
                britney_location, guard_location = self.pull_britney(
                    britney_location, guard_location
                    )
        else:
            guard_location, bump_reward = self.get_next_position(
                action, guard_location
                )
            reward -= float(bump_reward)
        britney_location = self.britney_stumbles(self.stumple_prob, britney_location)
    
        done = False
    
        if self.time_elapsed == self.time_limit:
            done = True
        
        if (britney_location == self.car_location).all():
            reward += self.size**2
            done = True
        
        self.britney_location = britney_location
        self.guard_location = guard_location
        state = self.states[guard_location[0]][guard_location[1]]\
            [britney_location[0]][britney_location[1]]
        return state, reward, done    
    
    def pull_britney(self, britney_location, guard_location):
        guard_location = britney_location.copy()
        
        x_change = 0
        y_change = 0
        
        if britney_location[0] == 1:
            x_change = 1
        if britney_location[0] == self.size-2:
            x_change = -1
        if britney_location[1] == 1:
            y_change = 1
        if britney_location[1] == self.size-2:
            y_change = -1
            
        britney_location[0] += x_change
        britney_location[1] += y_change
        
        return britney_location, guard_location

    def is_empty(self, position):
        """
        Helper function for get_next_position() to check if a call is empty, 
        i.e. no other agent or obstacle is on it
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
        """
        For every step, there is a probability that Britney moves randomly to an 
        adjacent location.
        """
        if stumble_probability >= random.random():
            action = random.choice(self.britney_actions)
            bl, _ = self.get_next_position(action, britney_location)
            return bl
        return britney_location
        
            
    def get_next_position(self, action, current_location):
        """
        Computes next location given some action and returns that location if
        it is empty. If it is not empty, e.g. if britney is on it, return 
        current location. 
        """
        bump_reward = 0
        if action == 'N': # North, i.e. up
            next_location = np.array((current_location[0] - 1, current_location[1]))
            if self.is_empty(next_location):
                return next_location, bump_reward    
        if action == 'S': # South, i.e. down
            next_location = np.array((current_location[0] + 1, current_location[1]))
            if self.is_empty(next_location):
                return next_location, bump_reward    
        if action == 'W': # West, i.e. left
            next_location = np.array((current_location[0], current_location[1] - 1))
            if self.is_empty(next_location):
                return next_location, bump_reward    
        if action == 'E': # East, i.e. right
            next_location = np.array((current_location[0], current_location[1] + 1))
            if self.is_empty(next_location):
                return next_location, bump_reward    
        if action == 'NE': # North east
            next_location = np.array((current_location[0] - 1, current_location[1] + 1))
            if self.is_empty(next_location):
                return next_location, bump_reward    
        if action == "SE": # South east
            next_location = np.array((current_location[0] + 1, current_location[1] + 1))
            if self.is_empty(next_location):
                return next_location, bump_reward    
        if action == "SW": # South west
            next_location = np.array((current_location[0] + 1, current_location[1] - 1))
            if self.is_empty(next_location):
                return next_location, bump_reward    
        if action == "NW": # North west
            next_location = np.array((current_location[0] - 1, current_location[1] -1))
            if self.is_empty(next_location):
                return next_location, bump_reward        
        
        bump_reward = 1
        return current_location, bump_reward
        
    def display(self):
        # Lifted from Michael's code
        
        envir_with_agent = np.asarray(self.map).copy()
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
        """
        This function resets the environment to a new initial state (time = 0).
        Then it places the agent and exit at new random locations.
        
        It is common practice to return the observations, 
        so that the agent can decide on the first action right after the 
        resetting of the environment.
        
        """
        self.time_elapsed = 0
        self.car_location = self.get_empty_cells(1)
        self.britney_start_location = self.get_empty_cells(1)
        self.britney_location = self.britney_start_location
        
        # The guard is spawned onto a location next to Britney, but cannot spawn
        # onto an obstacle
        britney_neighbors = list(self.get_neighbors(self.britney_location))
        britney_neighbors = [x for x in britney_neighbors if self.map[x[0]][x[1]] == 0]
        self.guard_start_location = np.asarray(random.choice(britney_neighbors))
        self.guard_location = self.guard_start_location
        
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
        

    def calculate_observations(self):
        relative_coordinates_car = self.car_location - self.britney_location
        relative_coordinates_britney = self.britney_location - self.guard_location
        surroundings= self.get_surroundings(
            self.guard_location, env_map=self.map, env_size=self.size, obs_size=self.obs_size)
        
        observation = {"relative_coordinates_car" : relative_coordinates_car,
                       "relative_coordinates_britney" : relative_coordinates_britney,
                       "surroundings" : surroundings}
        
        return observation
    
    def get_state(self):
        state = self.states[self.guard_location[0]][self.guard_location[1]]\
            [self.britney_location[0]][self.britney_location[1]]
        return state
    
    def get_surroundings(self, guard_location, env_map, env_size, obs_size):

        lower_bound = math.floor(obs_size / 2)
        upper_bound = lower_bound + 1 # +1 because slicing np.arrays is not inclusive
        
        row_start, row_end = guard_location[0] -lower_bound, guard_location[0] +upper_bound
        col_start, col_end = guard_location[1] -lower_bound, guard_location[1] +upper_bound
        
        pad = False
        
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0
        
        if row_start < 0:
            pad_top = abs(row_start)
            row_start = 0
            pad = True
        if row_end > env_size:
            pad_bottom = row_end-env_size
            row_end = env_size+1
            pad = True
        if col_start < 0:
            pad_left = abs(col_start)
            col_start = 0
            pad = True
        if col_end > env_size:
            pad_right = col_end-env_size
            col_end = env_size+1
            pad = True
         
        surroundings = env_map[row_start:row_end, col_start:col_end]
        
        if pad:
            surroundings = np.pad(
                np.array(surroundings),
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=-1
                )
            
            surroundings = torch.Tensor(surroundings)
            
        return surroundings

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
        o = observation['surroundings'].flatten()/norm
        state_tensor = np.concatenate([rel_coor_car, rel_coor_britney, o])
        #state_tensor = torch.tensor(state_tensor).unsqueeze(0)
        state_tensor = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0)
        
        return state_tensor
