# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:09:10 2021

@author: groes
"""
import torch 
import reinforcement_learning.task3.networks.q_network as qnet
import torch.nn.functional as F
import reinforcement_learning.task3.networks.policy_network as pi_net
import torch.nn as nn
from copy import deepcopy
import itertools
from torch.optim import Adam
import reinforcement_learning.task3.utils as utils
import numpy as np

class Actor_Critic(nn.Module):
    
    def __init__(self, num_obs, num_actions, hidden_sizes, activation_func):
        super().__init__()
        
        
        self.policy = pi_net.PiNet(num_obs, num_actions, hidden_sizes, activation_func, "popo")
        self.q1 = qnet.QNet(num_obs, num_actions, hidden_sizes, activation_func, "Q1")
        self.q2 = qnet.QNet(num_obs, num_actions, hidden_sizes, activation_func, "Q2")

class DiscreteSAC:
    """
    Soft Actor Critic algorithm designed to act in environment with discrete action space
    
    Adapted from: 
        https://arxiv.org/pdf/1910.07207.pdf
        https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
        
    """
    def __init__(self, ac_params, params):

        
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        

        
        ####################### ammended with spinning up ################
        self.actor_critic = Actor_Critic(
            ac_params['num_obs'],
            ac_params['num_actions'],
            ac_params['hidden_sizes'],
            ac_params['activation_func']
            )
        #a deep copy is for compound objects, it copies the object and then inserts copies of
        #its components
        #a shallow/normal copy copies and object then inserts references into the copy of
        #the objects in the origional
        self.target_actor_critic = deepcopy(self.actor_critic)
        
        #for efficient looping, just loops one after the other
        self.q_params = itertools.chain(self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters())
            
        self.pi_optimizer = Adam(self.actor_critic.policy.parameters(), lr=params["lr"])
        self.q_optimizer = Adam(self.q_params, lr=params["lr"]) 
        
        # Making new optimizers for the q net because I dont think we need to 
        # update a different set of parameters for each network, and I don't 
        # see how we can do that if all the params are stored in one object
        self.q1_optimizer = Adam(self.actor_critic.q1.parameters(), lr=params["lr"])
        self.q2_optimizer = Adam(self.actor_critic.q2.parameters(), lr=params["lr"])
        
        ###############################################################
    
    #def run(self, environment, policy, buffer, batchsize):
     #   Done = False
      #  while not Done:
       #     self.environment_step(environment, policy, buffer)
        #    self.gradient_step(buffer, batchsize)
        


        
    def environment_step(self, environment, buffer):
        # get state/observation from environment.
        observation = environment.calculate_observations()
        converted_obs = utils.convert_state(observation)
        # get action from state using policy.
        with torch.no_grad():
            action_distribuion = self.actor_critic.policy(converted_obs)
        action = np.max(np.array(action_distribuion))
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
        buffer.append(converted_obs, action, reward, converted_new_obs, done_num)
        
        #spinning up records dones as 1 if True and 0 if False
        
    
    def take_optimization_step(self, optimizer, network, loss, clipping_norm=None):
        optimizer.zero_grad()
        loss.backward()
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm(net.parameters(), clipping_norm)
        optimizer.step()
        
    def soft_update_of_target_net(self, target_model, local_model, tau):
        for target_param, local_param in zip(target_model.parameters(), 
                                             local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
    def gradient_step(self, buffer, batchsize, tau):
        """
        Args:
            tau (float) : used to regularize the soft update of the target nets
        """
        ############################### randomly sample a batch of transitions from D ############
        states, actions, rewards, new_states, dones = buffer.sample(batchsize)
        #compute the targets for Q functions
        #targets = self.compute_targets(states, new_states, actions, rewards, dones)
        
        ############################## update both local Qs with gradient descent #################
        q1_loss, q2_loss, q_min_loss = self.calc_q_loss(states, actions, rewards, new_states, dones)
        
        self.take_optimization_step(self.q1_optimizer, self.actor_critic.q1, q1_loss)
        self.take_optimization_step(self.q2_optimizer, self.actor_critic.q2, q2_loss)
        self.soft_update_of_target_net(self.target_actor_critic.q1, self.actor_critic.q1, tau)
        self.soft_update_of_target_net(self.target_actor_critic.q2, self.actor_critic.q2, tau)
        
        
        
        
        
        
        
        
        

        # is this how you do it?
        q_min_loss.backward()
        self.q_optimizer.step()
        
        ############################ update the policy with gradient ascent ########################
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        policy_loss = self.policy_loss(states, actions, rewards, new_states, dones)
        ########## apparently this should do it for gradient ascent
        policy_loss = - policy_loss
        policy_loss.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks 
        for p in self.q_params:
            p.requires_grad = True
            
        ######################### update target networks using polyak averaging ###################
        self.polyak_target_update(self.actor_critic, self.target_actor_critic)
        
    
        
        #with torch.no_grad():
         #   for p, p_targ in zip(self.actor_critic.parameters(), self.target_actor_critic.parameters()):
#
 #               p_targ.data.mul_(self.polyak)
  #              p_targ.data.add_((1 - self.polyak) * p.data)        
   


     
    def polyak_target_update(self, local_model, target_model):
        with torch.no_grad():
            for local_param, target_param in zip(self.actor_critic.parameters(), self.target_actor_critic.parameters()):

                target_param.data.mul_(self.polyak)
                target_param.data.add_((1 - self.polyak) * local_param.data)    
    
    def calc_q_loss(self, state_batch, action_batch, reward_batch, next_state_batch, dones_batch):
        """ Loss function for qnet """
        
        # Estimate target q_value
        with torch.no_grad():
            # Produce two q values
            q_next_target = self.target_actor_critic.q1(next_state_batch) # estimate via q_net
            #take max of this? since it outputs the qvalues for all actions given the observation
            q_next_target2 = self.target_actor_critic.q1(next_state_batch)
            #take max of this?
            qf_min = torch.min(q_next_target, q_next_target2)
            
            #the policy_net is from local ac as per the greek, 
            action_probabilities = self.actor_critic.policy(next_state_batch)#self.calc_action_prob() # TBD - it's the policy network
            log_action_probabilities = self.calc_log_prob(action_probabilities)
            
            # Calculate policy value
            v = action_probabilities * qf_min - self.alpha * log_action_probabilities
            v = v.sum(dim=1).unsqueeze(-1)
            
            # Dunno why (1.0 - dones_batch) is used, but he does it in his implementation
            # Answer: if next state then done, then the value of the move is = to rweard only

            target_q_value = reward_batch + (1.0 - dones_batch) + self.gamma * v 
        

        # Estimate q values with net and gather values
        # The qnets ouput a Q value for each action, so we use gather() to gather
        # the values corresponding to the action indices of the batch
        # explanation of gather() https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        q1 = self.actor_critic.q1(state_batch).gather(1, action_batch.long()) 
        q2 = self.actor_critic.q2(state_batch).gather(1, action_batch.long())
        
        q1_loss = F.mse_loss(q1, target_q_value)
        q2_loss = F.mse_loss(q2, target_q_value)
        return q1_loss, q2_loss#, qf_min
        
    def calc_log_prob(self, action_probabilities):
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        return torch.log(action_probabilities + z)
    
    #def calc_action_prob(self):
     #   # This is supposed to be the policy network
      #  pass
    

    
    
    def policy_loss(self, state_batch, action_batch, reward_batch, next_state_batch, dones_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        
        action_probabilities = self.actor_critic.policy(state_batch)
        log_action_probabilities = self.calc_log_prob(action_probabilities)
        
        
        q1 = self.actor_critic.q1(state_batch).gather(1, action_batch.long()) 
        q2 = self.actor_critic.q2(state_batch).gather(1, action_batch.long())
        
        min_q = torch.min(q1,q2)
        #action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        #qf1_pi = self.critic_local(state_batch)
        #qf2_pi = self.critic_local_2(state_batch)
        #_, _, min_qf_pi = self.calc_q_loss(state_batch, action_batch, reward_batch, next_state_batch, dones_batch)
        
       # action_probabilities = self.calc_action_prob()
        #log_action_probabilities = self.calc_log_prob(action_probabilities)
        
        
        inside_term = self.alpha * log_action_probabilities - min_q
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    