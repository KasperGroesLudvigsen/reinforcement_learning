# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:09:10 2021

@author: groes
"""
import torch 
import networks.q_network as qnet
import torch.nn.functional as F

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
        
        self.qnet_target = qnet.QNet(
            num_obs=params["num_obs"],
            num_actions=params["num_actions"],
            hidden_sizes=params["hidden_sizes"],
            activation_func=params["activation_func"],
            name=params["name"]
            )
        
        self.qnet_target2 = qnet.QNet(
            num_obs=params["num_obs"],
            num_actions=params["num_actions"],
            hidden_sizes=params["hidden_sizes"],
            activation_func=params["activation_func"],
            name=params["name"]
            )
        
        self.qnet_local = qnet.QNet(
            num_obs=params["num_obs"],
            num_actions=params["num_actions"],
            hidden_sizes=params["hidden_sizes"],
            activation_func=params["activation_func"],
            name=params["name"]
            )
        
        self.qnet_local2 = qnet.QNet(
            num_obs=params["num_obs"],
            num_actions=params["num_actions"],
            hidden_sizes=params["hidden_sizes"],
            activation_func=params["activation_func"],
            name=params["name"]
            )
        
    
    def calc_q_loss(self, state_batch, action_batch, reward_batch, next_state_batch, dones_batch):
        """ Loss function for qnet """
        
        # Estimate target q_value
        with torch.no_grad():
            # Produce two q values
            q_next_target = self.qnet_target(next_state_batch) # estimate via q_net
            q_next_target2 = self.qnet_target2(next_state_batch)
            qf_min = torch.min(q_next_target, q_next_target2)
    
            action_probabilities = self.calc_action_prob() # TBD - it's the policy network
            log_action_probabilities = self.calc_log_prob() # tbd
            
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
        
        return q1_loss, q2_loss
        
    def calc_log_prob(self, action_probabilities):
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        return torch.log(action_probabilities + z)
    
    def calc_action_prob(self):
        # This is supposed to be the policy network
        pass