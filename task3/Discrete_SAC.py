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

import utils.utils as utils
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
        
        #if torch.cuda.is_available():
        #    self.device = torch.cuda.device("cuda")
        #else: 
        #self.device = torch.cuda.device("cpu")

        # Hyperparameters
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.polyak = params["polyak"] # aka tau; regularizes target net updates
        self.clipping_norm = params["clipping_norm"]
        self.automatic_entropy_tuning = params["automatic_entropy_tuning"] # bool 
        
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
        
    def environment_step(self, environment, buffer):
        # get state/observation from environment.
        observation = environment.calculate_observations()
        converted_obs = utils.convert_state(observation)
        
        # get action from state using policy.
        with torch.no_grad():
            action_distribuion = self.actor_critic.policy(converted_obs)
        action = np.max(np.array(action_distribuion.squeeze()))
        # get reward, next state, done from environment by taking action in the world.
        _, reward, done = environment.take_action_guard(
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
        print(done_num)
        buffer.append(converted_obs, np.array(action_distribuion.squeeze()), reward, converted_new_obs, done_num)
        
        
    def gradient_step(self, buffer, batchsize):
        """
        This function performs the learning steps for all networks
        
        args:
            buffer (ReplayBuffer) : Contains past experiences (transitions)
            
            batchsizez (int) : how many transitions to train on before backprop
        """
        
        # Randomly sample a batch of transitions from replay buffer 
        states, actions, rewards, new_states, dones = buffer.sample(batchsize)

        # Update both local Qs with gradient descent 
        q1_loss, q2_loss = self.calc_q_loss(
            states, actions, rewards, new_states, dones
            )
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
        for p in self.q_params:
            p.requires_grad = False
        
        # Updating policy
        policy_loss, _ = self.calc_policy_loss(
            states, actions, rewards, new_states, dones
            )
        self.take_optimization_step(
            self.pi_optimizer, self.actor_critic.policy, 
            policy_loss, self.clipping_norm
            )
        
        # Unfreezing q nets while updating policy net
        for p in self.q_params:
            p.requires_grad = True
        
        # Perform learning step on entropy tuning parameter and update it
        if self.automatic_entropy_tuning:
            # Adapted from
            # https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/6297608b8524774c847ad5cad87e14b80abf69ce/agents/actor_critic_agents/SAC.py#L193 
            alpha_loss = self.calculate_entropy_tuning_loss()
            self.take_optimization_step(
                self.alpha_optim, None, alpha_loss, None
                )
            self.alpha = self.log_alpha.exp()


    def take_optimization_step(self, optimizer, network, loss,
                               clipping_norm=None):
        optimizer.zero_grad()
        loss.backward()
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm(net.parameters(), clipping_norm)
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
            q_next_target2 = self.target_actor_critic.q1(next_state_batch)
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
            
        # Estimate q values with q net and gather values
        # The qnets ouput a Q value for each action, so we use gather() to gather
        # the values corresponding to the action indices of the batch.
        # Explanation of gather() https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        q1 = self.actor_critic.q1(state_batch).gather(1, action_batch.long()) 
        q2 = self.actor_critic.q2(state_batch).gather(1, action_batch.long())
        
        q1_loss = F.mse_loss(q1[:,0], target_q_value.squeeze())
        q2_loss = F.mse_loss(q2[:,0], target_q_value.squeeze())
        #q1_loss = F.mse_loss(q1, target_q_value)
        #q2_loss = F.mse_loss(q2, target_q_value)
        return q1_loss, q2_loss
        
        
    
    def calc_log_prob(self, action_probabilities):
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        return torch.log(action_probabilities + z)
    
    

    def calc_policy_loss(self, state_batch):
        """
        Calculates the loss for the actor. This loss includes the additional
        entropy term
        """

        action_probabilities = self.actor_critic.policy(state_batch)
        log_action_probabilities = self.calc_log_prob(action_probabilities)
        
        # Estimate q values and get min to be used in inside_term
        q1 = self.actor_critic.q1(state_batch)
        q2 = self.actor_critic.q2(state_batch)

        min_q = torch.min(q1,q2)
        
        # Calculate log_action_probabilities
        inside_term = self.alpha * log_action_probabilities - min_q
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()

        return policy_loss
    
    
    def calc_entropy_tuning_loss(self, log_pi):
        # TBD
        pass
    
    def polyak_target_update(self, local_model, target_model):
        """ 
            We're not using this method so we can get rid of it if the soft_update works
        """
        with torch.no_grad():
            for local_param, target_param in zip(self.actor_critic.parameters(), self.target_actor_critic.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1 - self.polyak) * local_param.data)    
    
    
    def calculate_entropy_tuning_loss(self, log_pi):
        # Copied from: 
        # https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/6297608b8524774c847ad5cad87e14b80abf69ce/agents/actor_critic_agents/SAC.py#L181
        """Calculates the loss for the entropy temperature parameter. This is 
        only relevant if self.automatic_entropy_tuning is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss
    
    
    
    
    
    
    
    
    
    