""" Learn a policy using DDPG for the reach task"""
import numpy as np
import torch
import torch.nn.functional as F
import copy
from torch.distributions import MultivariateNormal

from utils import weighSync
from models import Actor, Critic
from replay import Replay


class DDPG():
    def __init__(
            self,
            env,
            state_dim = 8,
            action_dim = 2,
            critic_lr=3e-4,
            actor_lr=3e-4,
            gamma=0.99,
            batch_size=100,
    ):
        """
        param: env: An gym environment
        param: action_dim: Size of action space
        param: state_dim: Size of state space
        param: critic_lr: Learning rate of the critic
        param: actor_lr: Learning rate of the actor
        param: gamma: The discount factor
        param: batch_size: The batch size for training
        """
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env

        self.actor = Actor(state_dim, action_dim)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr= actor_lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr= critic_lr)

        self.RB = Replay(10000, 1000, state_dim, action_dim, env)
        
        self.avr = []

    def update_target_networks(self):
        """
        A function to update the target networks
        """
        weighSync(self.actor_target, self.actor)
        weighSync(self.critic_target, self.critic)

    def update_network(self):
        """
        A function to update the function just once
        """
        self.optimizer_critic.zero_grad()
        self.critic_loss.backward()
        self.optimizer_critic.step()
        
        self.optimizer_actor.zero_grad()
        self.actor_loss.backward()
        self.optimizer_actor.step()

    def train(self, num_steps):
        """
        Train the policy for the given number of iterations
        :param num_steps:The number of steps to train the policy for
        """
        obs = self.env.reset()
        num = 0
        summ = 0
        t = 0
        r = []
        for i in range(num_steps):
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            mu = self.actor(obs)
            m = MultivariateNormal(mu, covariance_matrix = 0.1 * torch.eye(2))
            action = m.sample()
            action = np.clip(action[0].numpy(), -1, 1)
            nobs, reward, done, _ = self.env.step(action)
            self.RB.add((obs.numpy().reshape(-1), action, reward, nobs, done))
            
            num += 1
            summ += 1
            r.append(reward)
            
            # update
            states, actions, rewards, next_states, dones = self.RB.sample(self.batch_size)
            next_actions = self.actor_target(next_states)
            Q_target = self.critic_target(next_states, next_actions)
            y = rewards + self.gamma * (1 - dones) * Q_target
            # update models
            self.critic_loss = F.mse_loss(self.critic(states, actions), y)
            self.actor_loss = - self.critic(states, self.actor(states)).mean()

            self.update_network()
            self.update_target_networks()
            
            if done: 
                avr = sum(r)
                self.avr.append(avr)
                print('Training %.2f%%, %d episodes, number of actions:%d, returns: %.3f'%(summ/num_steps * 100, t, num, avr))
                
                obs = self.env.reset()
                t += 1
                num = 0
                r = []
            else:
                obs = nobs

                          
                
                
                





