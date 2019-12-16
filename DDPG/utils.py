# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 12:54:53 2019

@author: user
"""
import time
import torch

def valuate(env, policy):
    num_iter = 300
    num_try = 149
    reached = 0
    for i in range(num_iter):
        obs = env.reset()
        for j in range(num_try):
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            action = policy(obs)
            obs, reward, done, _ = env.step(action.detach().squeeze().numpy())
            if done:
                reached += 1
                break
    return reached

def tic():
    return time.time()

def toc(x):
    return time.time() - x

def weighSync(target_model, source_model, tau=0.001):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(tau*source_param.data + (1.0-tau)*target_param.data)

def evaluate_policy(env, policy):
    obs = env.reset()
    env.render('human')
    steps = 0
    done = False
    time.sleep(3)
    while steps<10000:
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        
        action = policy(obs).detach().squeeze().numpy()
        obs, r, done, info = env.step(action)
        if done:
            obs = env.reset()
        steps+=1
        env.render('human')
        time.sleep(0.1)