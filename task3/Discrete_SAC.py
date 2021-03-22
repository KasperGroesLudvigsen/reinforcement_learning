# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:09:10 2021

@author: groes
"""
import torch 
import networks.q_network as qnet
import torch.nn.functional as F
import networks.policy_network as pi_net
import torch.nn as nn
from copy import deepcopy

class Actor_Critic(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        
        self.policy = pi_net()
        self.q1 = qnet()
        self.q2 = qnet()

class DiscreteSAC:
    """
    Soft Actor Critic algorithm designed to act in environment with discrete action space
    
    Adapted from: 
        https://arxiv.org/pdf/1910.07207.pdf
        https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
        
    """
    def __init__(self, params):

        
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        
        #self.qnet_target = qnet.QNet(
        #    num_obs=params["num_obs"],
        #    num_actions=params["num_actions"],
        #    hidden_sizes=params["hidden_sizes"],
        #    activation_func=params["activation_func"],
        #    name=params["name"]
        #    )
        
        #self.qnet_target2 = qnet.QNet(
        #    num_obs=params["num_obs"],
        #    num_actions=params["num_actions"],
        #    hidden_sizes=params["hidden_sizes"],
        #    activation_func=params["activation_func"],
        #    name=params["name"]
        #    )
        
        #self.qnet_local = qnet.QNet(
        #    num_obs=params["num_obs"],
        #    num_actions=params["num_actions"],
        #    hidden_sizes=params["hidden_sizes"],
        #    activation_func=params["activation_func"],
        #    name=params["name"]
        #    )
        
        #self.qnet_local2 = qnet.QNet(
        #    num_obs=params["num_obs"],
        #    num_actions=params["num_actions"],
        #    hidden_sizes=params["hidden_sizes"],
        #    activation_func=params["activation_func"],
        #    name=params["name"]
        #    )
        
        # Target networks are instantiated to be a copy of local networks, as per spinning up
        # pseudo code
        
        #self.qnet_target = self.qnet_local
        #self.qnet_target2 = self.qnet_local2
        
        ####################### ammended with spinning up ################
        self.actor_critic = Actor_Critic()
        self.target_actor_critic = deepcopy(self.actor_critic)
    
        self.q_params = itertools.chain(self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters())
            
        self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(q_params, lr=lr)        
        ###############################################################
    
    def run(self):
        while not Done:
            self.environment_step(environment, policy, buffer)
            self.gradient_step(buffer, batchsize)
        


        
    def environment_step(self, environment, policy, buffer):
        # get state/observation from environment.
        state = environment.get_state()
        # get action from state using policy.
        action = policy.get_action(state)
        # get reward, next state, done from environment by taking action in the world.
        new_state, reward, done = environment.take_step(action)
        # store the experience in D, experience replay buffer.
        if done:
            done_num = 1
        else:
            done_num = 0
        buffer.append(state, action, reward, new_state, done_num)
        
        #spinning up records dones as 1 if True and 0 if False
        
    
    def gradient_step(self, buffer, batchsize):
        ############################### randomly sample a batch of transitions from D ############
        states, new_states, actions, rewards, dones = buffer.sample(batchsize)
        #compute the targets for Q functions
        #targets = self.compute_targets(states, new_states, actions, rewards, dones)
        
        ############################## update both local Qs with gradient descent #################
        self.q_optimiser.zero_grad()
        q1_loss, q2_loss, qf_min = self.calc_q_loss(states, new_states, actions, rewards, dones)
        # is this how you do it?
        qf_min.backward()
        self.q_optimiser.step()
        
        ############################ update the policy with gradient ascent ########################
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimiser.zero_grad()
        policy_loss = self.policy_loss(states, new_states, actions, rewards, dones)
        policy_loss.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks 
        for p in self.q_params:
            p.requires_grad = True
            
        ######################### update target networks using polyak averaging ###################
        with torch.no_grad():
            for p, p_targ in zip(self.actor_critic.parameters(), self.target_actor_critic.parameters()):

                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)        
        
    
    def calc_q_loss(self, state_batch, action_batch, reward_batch, next_state_batch, dones_batch):
        """ Loss function for qnet """
        
        # Estimate target q_value
        with torch.no_grad():
            # Produce two q values
            q_next_target = self.qnet_target(next_state_batch) # estimate via q_net
            #take max of this? since it outputs the qvalues for all actions given the observation
            q_next_target2 = self.qnet_target2(next_state_batch)
            #take max of this?
            qf_min = torch.min(q_next_target, q_next_target2)
    
            action_probabilities = self.actor_critic.pi(next_state_batch)#self.calc_action_prob() # TBD - it's the policy network
            log_action_probabilities = self.calc_log_prob(action_probabilities) # tbd
            
            # Calculate policy value
            v = action_probabilities * qf_min - self.alpha * log_action_probabilities
            v = v.sum(dim=1).unsqueeze(-1)
            
            # Dunno why (1.0 - dones_batch) is used, but he does it in his implementation
            target_q_value = reward_batch + (1.0 - dones_batch) + self.gamma * v 
        

        # Estimate q values with net and gather values
        # The qnets ouput a Q value for each action, so we use gather() to gather
        # the values corresponding to the action indices of the batch
        # explanation of gather() https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        q1 = self.qnet_local(state_batch).gather(1, action_batch.long()) 
        q2 = self.qnet_local2(state_batch).gather(1, action_batch.long())
        
        q1_loss = F.mse_loss(q1, target_q_value)
        q2_loss = F.mse_loss(q2, target_q_value)
        return q1_loss, q2_loss, qf_min
        
    def calc_log_prob(self, action_probabilities):
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        return torch.log(action_probabilities + z)
    
    def calc_action_prob(self):
        # This is supposed to be the policy network
        pass
    

    
    
    def policy_loss(self, state_batch, action_batch, reward_batch, next_state_batch, dones_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        #action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        #qf1_pi = self.critic_local(state_batch)
        #qf2_pi = self.critic_local_2(state_batch)
        _, _, min_qf_pi = self.calc_q_loss(state_batch, action_batch, reward_batch, next_state_batch, dones_batch)
        
        action_probabilities = self.calc_action_prob()
        log_action_probabilities = self.calc_log_prob(action_probabilities)
        
        
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    