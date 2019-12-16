# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:06:06 2019

@author: user
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the network
        param: state_dim : Size of the state space
        param: action_dim: Size of the action space
        """
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, 400)
        self.linear1.weight.data.uniform_(-1/math.sqrt(state_dim), 1/math.sqrt(state_dim))
        self.linear2 = nn.Linear(400, 300)
        self.linear2.weight.data.uniform_(-1/math.sqrt(400), 1/math.sqrt(400))
        self.linear3 = nn.Linear(300, action_dim)
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Define the forward pass
        param: state: The state of the environment
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return torch.tanh(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the critic
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        """
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_dim, 400)
        self.linear1.weight.data.uniform_(-1/math.sqrt(state_dim), 1/math.sqrt(state_dim))
        self.linear2 = nn.Linear(400 + action_dim, 300)
        self.linear2.weight.data.uniform_(-1/math.sqrt(400 + action_dim), 1/math.sqrt(400))
        self.linear3 = nn.Linear(300, 1)
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Define the forward pass of the critic
        """
        x = F.relu(self.linear1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x