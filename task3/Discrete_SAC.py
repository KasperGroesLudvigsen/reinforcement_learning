# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:09:10 2021

@author: groes
"""
import torch 
import task3.networks.q_network as qnet
import torch.nn.functional as F
import task3.networks.policy_network as pi_net
import torch.nn as nn
from copy import deepcopy
import itertools
from torch.optim import Adam
import random

import task3.utils as utils
#import reinforcement_learning.utils.utils as utils

import numpy as np

class Actor_Critic(nn.Module):
    """
    Contains the policy net and the two q nets. 
    
    """
    def __init__(self, num_obs, num_actions, hidden_sizes, activation_func):
        super().__init__()
        
        self.policy = pi_net.PiNet(num_obs, num_actions, hidden_sizes, activation_func, "popo")
        self.q1 = qnet.QNet(num_obs, num_actions, hidden_sizes, activation_func, "Q1")
        self.q2 = qnet.QNet(num_obs, num_actions, hidden_sizes, activation_func, "Q2")


class DiscreteSAC:
    """
    Soft Actor Critic algorithm designed to act in environment with discrete action space
    
    Inspired by: 
        https://arxiv.org/abs/1801.01290 
        https://arxiv.org/pdf/1910.07207.pdf
        https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
        https://github.com/openai/spinningup/tree/038665d62d569055401d91856abb287263096178
        
    """
    def __init__(self, ac_params, params):
        
        if torch.cuda.is_available():
            self.device = torch.cuda.device("cuda")
        else: 
            self.device = torch.device("cpu")

        # Hyperparameters
        self.gamma = params["gamma"]
        self.polyak = params["polyak"] # aka tau; regularizes target net updates
        self.clipping_norm = params["clipping_norm"]
        self.automatic_entropy_tuning = params["automatic_entropy_tuning"] # bool 
        self.train_mode = True
        
        # Actor critic object contains 2 q nets and the policy net
        self.actor_critic = Actor_Critic(
            ac_params['num_obs'],
            ac_params['num_actions'],
            ac_params['hidden_sizes'],
            ac_params['activation_func']
            )
        
        # A deep copy is for compound objects, it copies the object and then 
        # inserts copies of its components
        # A shallow/normal copy copies and object then inserts references into
        # the copy of the objects in the origional
        self.target_actor_critic = deepcopy(self.actor_critic)
        
        # Freeze target networks with respect to optimizers (only update via 
        # polyak averaging)
        for p in self.target_actor_critic.parameters():
            p.requires_grad = False
        
        # For efficient looping, just loops one after the other
        self.q_params = itertools.chain(self.actor_critic.q1.parameters(),
                                        self.actor_critic.q2.parameters())
        
        self.pi_optimizer = Adam(self.actor_critic.policy.parameters(), lr=params["lr"])
        self.q1_optimizer = Adam(self.actor_critic.q1.parameters(), lr=params["lr"])
        self.q2_optimizer = Adam(self.actor_critic.q2.parameters(), lr=params["lr"])
        
        # Setting entropy tuning parameter (alpha)
        if self.automatic_entropy_tuning:
            # Copied from:
            # https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/6297608b8524774c847ad5cad87e14b80abf69ce/agents/actor_critic_agents/SAC.py#L189
            self.target_entropy = -torch.prod(torch.Tensor(ac_params['num_actions']).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=params["lr"], eps=1e-4)
        else:
            self.alpha = params["entropy_alpha"]
        
    def environment_step(self, environment, buffer, buffer_fill):
        # get state/observation from environment.
        observation = environment.calculate_observations()
        converted_obs = utils.convert_state(observation)
        #print(converted_obs)
        actions = ["N", "E", "S", "W", "NE", "SE", "SW", "NW" ,"push", "pull"]
        # get action from state using policy.
        if buffer_fill == False:
            with torch.no_grad():
                action_distribution = self.actor_critic.policy(converted_obs)
                #print('Action dis: ', action_distribution)
                #action_distribution = action_distribution.squeeze()
                #print('Action dis: ', action_distribution)
                #action_distribution = action_distribution.flatten().tolist()
                action_distribution = np.asarray(action_distribution)[0]
                #action_distribution = [x / sum(action_distribution) for x in action_distribution]
                if self.train_mode:    
                    action = np.random.choice(actions, p = action_distribution)
                    action_index = actions.index(action)
                else:
                    action_index = np.argmax(action_distribution)
                    action = actions[action_index]
            
        if buffer_fill == True:
            greedy_prob = random.random() # this should be between 0 and 1, 
            if greedy_prob < 1.0:
                action_index = random.randint(0,9)
                action = actions[action_index]
            else:
                with torch.no_grad():
                    #print('Action selection')
                    action_distribution = self.actor_critic.policy(converted_obs)
                    #action_distribution = action_distribution.squeeze()
                    #print('Action dis: ', action_distribution.sum())
                    #print('Action dis: ', action_distribution)
                    #action_distribution = action_distribution.flatten().tolist()
                    action_distribution = np.asarray(action_distribution)[0]
                    #action_distribution = [x / sum(action_distribution) for x in action_distribution]
                    if self.train_mode:    
                        action = np.random.choice(actions, p = action_distribution)
                        action_index = actions.index(action)
                    else:
                        action_index = np.argmax(action_distribution)
                        action = actions[action_index]
        
        
        
        # get reward, next state, done from environment by taking action in the world.
        reward, done = environment.environment_step(
            environment.guard_location,
            environment.britney_location,
            action)
        new_obs = environment.calculate_observations()
        converted_new_obs = utils.convert_state(new_obs)
        # store the experience in D, experience replay buffer.
        if done:
            done_num = 1
        else:
            done_num = 0
        #print('------------------------')   
        #print('initial state:', converted_obs)
        #print('action', action)
        #print('new state: ', converted_new_obs)
        buffer.append(converted_obs, action_index, reward, converted_new_obs, done_num)
        return done, reward
        
    def gradient_step(self, buffer, batchsize):
        """
        This function performs the learning steps for all networks
        
        args:
            buffer (ReplayBuffer) : Contains past experiences (transitions)
            
            batchsizez (int) : how many transitions to train on before backprop
        """
        
        # Randomly sample a batch of transitions from replay buffer 
        states, new_states, actions, rewards, dones = buffer.sample(batchsize)
        #print(states.shape, new_states.shape)
        states = states.squeeze()
        #print(states.shape, new_states.shape)
        new_states = new_states.squeeze()
        # Update both local Qs with gradient descent 
        q1_loss, q2_loss, logpi = self.calc_q_loss(
            states, actions, rewards, new_states, dones
            )
        #q1_loss = q1_loss.requires_grad_(True)
        #q2_loss = q2_loss.requires_grad_(True)
        self.take_optimization_step(
            self.q1_optimizer, self.actor_critic.q1, q1_loss, self.clipping_norm
            )
        self.take_optimization_step(
            self.q2_optimizer, self.actor_critic.q2, q2_loss, self.clipping_norm
            )
        
        # Soft updating target q nets
        self.soft_update_of_target_net(
            self.target_actor_critic.q1, self.actor_critic.q1, self.polyak
            )
        self.soft_update_of_target_net(
            self.target_actor_critic.q2, self.actor_critic.q2, self.polyak
            )
        
        # Freezing q nets while updating policy net
        #for p in self.q_params:
         #   p.requires_grad = False
        for p in self.actor_critic.q1.parameters():
            p.requires_grad = False
        for p in self.actor_critic.q2.parameters():
            p.requires_grad = False
        # Updating policy
        policy_loss = self.calc_policy_loss(states)
        #policy_loss = policy_loss.clone().detach().requires_grad_(True)
        self.take_optimization_step(
            self.pi_optimizer, self.actor_critic.policy, 
            policy_loss, self.clipping_norm
            )
        
        if self.automatic_entropy_tuning:
            # Adapted from
            # https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/6297608b8524774c847ad5cad87e14b80abf69ce/agents/actor_critic_agents/SAC.py#L193 
            alpha_loss = self.calculate_entropy_tuning_loss(logpi)
            self.take_optimization_step(
                self.alpha_optim, None, alpha_loss, None
                )
            self.alpha = self.log_alpha.exp()
        
        # Unfreezing q nets after updating policy net
        #for p in self.q_params:
         #   p.requires_grad = True
        for p in self.actor_critic.q1.parameters():
            p.requires_grad = True
        for p in self.actor_critic.q2.parameters():
            p.requires_grad = True
            
    
    def gradient_step_ere(self, states, new_states, actions, rewards, dones):
        """
        This function performs the learning steps for all networks
        
        args:
            buffer (ReplayBuffer) : Contains past experiences (transitions)
            
            batchsizez (int) : how many transitions to train on before backprop
        """
        
        # Randomly sample a batch of transitions from replay buffer 
        #states, new_states, actions, rewards, dones = buffer.sample(batchsize)
        #print(states.shape, new_states.shape)
        states = states.squeeze()
        #print(states.shape, new_states.shape)
        new_states = new_states.squeeze()
        # Update both local Qs with gradient descent 
        q1_loss, q2_loss, logpi = self.calc_q_loss(
            states, actions, rewards, new_states, dones
            )
        #q1_loss = q1_loss.requires_grad_(True)
        #q2_loss = q2_loss.requires_grad_(True)
        self.take_optimization_step(
            self.q1_optimizer, self.actor_critic.q1, q1_loss, self.clipping_norm
            )
        self.take_optimization_step(
            self.q2_optimizer, self.actor_critic.q2, q2_loss, self.clipping_norm
            )
        
        # Soft updating target q nets
        self.soft_update_of_target_net(
            self.target_actor_critic.q1, self.actor_critic.q1, self.polyak
            )
        self.soft_update_of_target_net(
            self.target_actor_critic.q2, self.actor_critic.q2, self.polyak
            )
        
        # Freezing q nets while updating policy net
        #for p in self.q_params:
         #   p.requires_grad = False
        for p in self.actor_critic.q1.parameters():
            p.requires_grad = False
        for p in self.actor_critic.q2.parameters():
            p.requires_grad = False
        # Updating policy
        policy_loss = self.calc_policy_loss(states)
        #policy_loss = policy_loss.clone().detach().requires_grad_(True)
        self.take_optimization_step(
            self.pi_optimizer, self.actor_critic.policy, 
            policy_loss, self.clipping_norm
            )
        
        if self.automatic_entropy_tuning:
            # Adapted from
            # https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/6297608b8524774c847ad5cad87e14b80abf69ce/agents/actor_critic_agents/SAC.py#L193 
            alpha_loss = self.calculate_entropy_tuning_loss(logpi)
            self.take_optimization_step(
                self.alpha_optim, None, alpha_loss, None
                )
            self.alpha = self.log_alpha.exp()
        
        # Unfreezing q nets after updating policy net
        #for p in self.q_params:
         #   p.requires_grad = True
        for p in self.actor_critic.q1.parameters():
            p.requires_grad = True
        for p in self.actor_critic.q2.parameters():
            p.requires_grad = True
            
    
        # Perform learning step on entropy tuning parameter and update it

            
    #def gradient_step_experiment(self, buffer, batchsize):
    #    """
    #    This function performs the learning steps for all networks
    #    
    #    args:
    #        buffer (ReplayBuffer) : Contains past experiences (transitions)
    #        
    #        batchsizez (int) : how many transitions to train on before backprop
    #    """
    #    
    #    # Randomly sample a batch of transitions from replay buffer 
    #    states, new_states, actions, rewards, dones = buffer.sample(batchsize)
    #    #print(states.shape, new_states.shape)
    #    states = states.squeeze()
    #    #print(states.shape, new_states.shape)
    #    new_states = new_states.squeeze()
    #    # Update both local Qs with gradient descent 
    #    self.q1_optimizer.zero_grad()
    #    self.q2_optimizer.zero_grad()
    #    q1_loss, q2_loss, logpi = self.calc_q_loss(
    #        states, actions, rewards, new_states, dones
    #        )
    #    #q1_loss = q1_loss.requires_grad_(True)
    #    #q2_loss = q2_loss.requires_grad_(True)
    #    q1_loss.backward
    #   q2_loss.backward
    #   
    #    self.q1_optimizer.step()
    #    self.q2_optimizer.step()

        
        # Soft updating target q nets
    #    self.soft_update_of_target_net(
    #        self.target_actor_critic.q1, self.actor_critic.q1, self.polyak
    #        )
    #    self.soft_update_of_target_net(
    #        self.target_actor_critic.q2, self.actor_critic.q2, self.polyak
    #        )
        
        # Freezing q nets while updating policy net
    #    for p in self.q_params:
    #        p.requires_grad = False
    #    
    #    # Updating 
    #    self.pi_optimizer.zero_grad()
    #    policy_loss = self.calc_policy_loss(states)
    #    #policy_loss = policy_loss.clone().detach().requires_grad_(True)
    #    policy_loss = - policy_loss
    #    policy_loss.backward
    #    self.pi_optimizer.step()
    #    
        # Unfreezing q nets while updating policy net
    #    for p in self.q_params:
    #        p.requires_grad = True
    #    
        # Perform learning step on entropy tuning parameter and update it
    #    if self.automatic_entropy_tuning:
    #        # Adapted from
    #        # https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/6297608b8524774c847ad5cad87e14b80abf69ce/agents/actor_critic_agents/SAC.py#L193 
    #        alpha_loss = self.calculate_entropy_tuning_loss(logpi)
    #        self.take_optimization_step(
    #            self.alpha_optim, None, alpha_loss, None
    #            )
    #        self.alpha = self.log_alpha.exp()

    def take_optimization_step(self, optimizer, network, loss,
                               clipping_norm=None):
        optimizer.zero_grad()
        loss.backward()
        if clipping_norm is not None:
            for param in network.parameters():
                torch.nn.utils.clip_grad_norm(param, clipping_norm)
        optimizer.step()
        
        
    def soft_update_of_target_net(self, target_model, local_model, tau):
        """ 
            Polyak smoothing / averaging used to update target q nets
        """
        for target_param, local_param in zip(target_model.parameters(), 
                                             local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data
                )     
    
    def calc_q_loss(self, state_batch, action_batch, reward_batch,
                    next_state_batch, dones_batch):
        """ Loss function for qnet """
        
        # Estimate target q_value
        with torch.no_grad():
            # Produce two q values via target q net and get min
            q_next_target = self.target_actor_critic.q1(next_state_batch) 
            q_next_target2 = self.target_actor_critic.q2(next_state_batch)
            qf_min = torch.min(q_next_target, q_next_target2)
            
            # The policy_net is from local ac
            action_probabilities = self.actor_critic.policy(next_state_batch)
            log_action_probabilities = self.calc_log_prob(action_probabilities)
            #print("action prob shape: {}".format(log_action_probabilities.shape))
            # Calculate policy value
            v = action_probabilities * (qf_min - self.alpha * log_action_probabilities)
            #print("v shape before squeeze: {}".format(v.shape))
            v = v.sum(dim=1).unsqueeze(-1)

            
            # Subtracting dones from 1 so that if next state is done, then the 
            # value of the move is = to rweard only
            reward_batch = reward_batch.unsqueeze(-1)
            mask = (1.0 - dones_batch).unsqueeze(-1)
            target_q_value = reward_batch + mask * self.gamma * v 
            #print("target q shape:",target_q_value.shape )
            
        # Estimate q values with q net and gather values
        # The qnets ouput a Q value for each action, so we use gather() to gather
        # the values corresponding to the action indices of the batch.
        action_batch.requires_grad_(True)
        # Explanation of gather() https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        q1 = self.actor_critic.q1(state_batch).gather(1, action_batch.unsqueeze(-1).long()) 
        q2 = self.actor_critic.q2(state_batch).gather(1, action_batch.unsqueeze(-1).long())

        q1_loss = F.mse_loss(q1, target_q_value)
        q2_loss = F.mse_loss(q2, target_q_value)
        return q1_loss, q2_loss, log_action_probabilities
        
        
    
    def calc_log_prob(self, action_probabilities):
        z = action_probabilities #== 0.0
        z = z.float() * 1e-9
        return torch.log(action_probabilities + z)
    
    

    def calc_policy_loss(self, state_batch):
        """
        Calculates the loss for the actor. This loss includes the additional
        entropy term
        """
        state_batch.requires_grad_(True)
        action_probabilities = self.actor_critic.policy(state_batch)
        log_action_probabilities = self.calc_log_prob(action_probabilities)
        
        # Estimate q values and get min to be used in inside_term
        with torch.no_grad():
            q1 = self.actor_critic.q1(state_batch)
            q2 = self.actor_critic.q2(state_batch)

        min_q = torch.min(q1,q2)
        
        # Calculate log_action_probabilities
        inside_term = self.alpha * log_action_probabilities - min_q
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()

        return policy_loss
    
    
    #def polyak_target_update(self, local_model, target_model):
     #   """ 
      #      We're not using this method so we can get rid of it if the soft_update works
       # """
        #with torch.no_grad():
         #   for local_param, target_param in zip(self.actor_critic.parameters(), self.target_actor_critic.parameters()):
         #       target_param.data.mul_(self.polyak)
          #      target_param.data.add_((1 - self.polyak) * local_param.data)    
    
    
    def calculate_entropy_tuning_loss(self, log_pi):
        # Copied from: 
        # https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/6297608b8524774c847ad5cad87e14b80abf69ce/agents/actor_critic_agents/SAC.py#L181
        """Calculates the loss for the entropy temperature parameter. This is 
        only relevant if self.automatic_entropy_tuning is True."""
        log_pi.requires_grad_(False)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        print(alpha_loss)
        return alpha_loss
    
    
    
    
    
    
    
    
    
    