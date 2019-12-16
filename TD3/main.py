# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:23:05 2019

@author: user
"""
import gym
import torch
from utils import tic, toc
import matplotlib.pyplot as plt
from td3 import TD3
import time

import argparse

parser = argparse.ArgumentParser()

env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init = False) # rand_init = False
seed = 789 #987
env.seed(seed)
torch.manual_seed(seed)

td3_object = TD3(
    env,
    8,
    2,
    critic_lr=1e-3,
    actor_lr=1e-3, 
    gamma=0.99,
    batch_size=100,
)
# Train the policy
td3_object.train(200000)

plt.plot(td3_object.avr)
plt.title(f'returns after every update when seed = {seed}')

## Evaluate the final policy
env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1")
steps = 0
env.render('human')
obs = env.reset()
done = False
time.sleep(3)
while steps<10000:
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    
    action = td3_object.actor(obs).detach().squeeze().numpy()
    obs, r, done, info = env.step(action)
    if done:
        obs = env.reset()
    steps+=1
    env.render('human')
    time.sleep(0.1)