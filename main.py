# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:10:14 2019

@author: user
"""

import gym
import torch
from utils import tic, toc
import matplotlib.pyplot as plt
from DDPG import DDPG
import time


env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init = False) # rand_init = False
seed = 789 #456, 789
env.seed(seed)
torch.manual_seed(seed)

ddpg_object = DDPG(
    env,
    8,
    2,
    critic_lr=1e-3,
    actor_lr=3e-4, # -4
    gamma=0.99,
    batch_size=64,
)
# Train the policy
print(f'Training with seed: {seed}')
a = tic()
ddpg_object.train(200000)
b = toc(a)
print(f'training time: {b}')

plt.plot(ddpg_object.avr)
plt.title(f'returns after every update when seed = {seed}')

## Evaluate the final policy
env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1")
steps = 0
env.render('human')
obs = env.reset()
done = False
time.sleep(3)
while steps<1000:
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    
    action = ddpg_object.actor(obs).detach().squeeze().numpy()
    obs, r, done, info = env.step(action)
    if done:
        obs = env.reset()
    steps+=1
    env.render('human')
    time.sleep(0.1)