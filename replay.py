# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:08:25 2019

@author: user
"""
from collections import deque
import random
import numpy as np
import torch

class Replay():
    def __init__(self, buffer_size, init_length, state_dim, action_dim, env):
        """
        A function to initialize the replay buffer.

        param: init_length : Initial number of transitions to collect
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        param: env : gym environment object
        """
        self.buffer_size = buffer_size
        self.init_length = init_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env = env
        
        self.memory = deque(maxlen = buffer_size)
        
        obs = env.reset()
        for i in range(1000):
            a, b = random.uniform(-1, 1), random.uniform(-1, 1)
            nobs, reward, done, _ = env.step([a, b])
            self.memory.append((obs, np.array([a, b]), reward, nobs, done))
            if done:
                obs = env.reset()
            else:
                obs = nobs

    def add(self, exp):
        """
        A function to add a dictionary to the buffer
        param: exp : A tuple consisting of state, action, reward , next state and done flag
        """
        self.memory.append(exp)

    def sample(self, N):
        """
        A function to sample N points from the buffer
        param: N : Number of samples to obtain from the buffer
        """
        sam = random.sample(self.memory, N)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for s in sam:
            states.append(s[0])
            actions.append(s[1])
            rewards.append(s[2])
            next_states.append(s[3])
            dones.append(s[4])
        states = torch.tensor(states).float()
        actions = torch.tensor(actions).float()
        rewards = torch.tensor(rewards).float().view(-1,1)
        next_states = torch.tensor(next_states).float()
        dones = torch.tensor(dones).float().view(-1,1)
        return states, actions, rewards, next_states, dones