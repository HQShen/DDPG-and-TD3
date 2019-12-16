import numpy as np
import torch
import torch.nn.functional as F
import copy
from torch.distributions import MultivariateNormal

from utils import weighSync
from models import Actor, Critic
from replay import Replay
        
        
class TD3():
    def __init__(
            self,
            env,
            state_dim = 8,
            action_dim = 2,
            critic_lr=3e-4,
            actor_lr=3e-4,
            gamma=0.99,
            batch_size=100,
            policy_noise = 0.1,
            noise_clip = 0.5,
            policy_delay = 2
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
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise

        # TODO: Create a actor and actor_target
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        # TODO: Make sure that both networks have the same initial weights

        # TODO: Create a critic and critic_target object
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        # TODO: Make sure that both networks have the same initial weights

        # TODO: Define the optimizer for the actor
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr= actor_lr)
        # TODO: Define the optimizer for the critic
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr= critic_lr)

        # TODO: define a replay buffer
        self.RB = Replay(10000, 1000, state_dim, action_dim, env)
        
        self.avr = []

    # TODO: Complete the function
    def update_target_networks(self):
        """
        A function to update the target networks
        """
        weighSync(self.actor_target, self.actor)
        weighSync(self.critic_target, self.critic)

    # TODO: Complete the function
    def train(self, num_steps):
        """
        Train the policy for the given number of iterations
        :param num_steps:The number of steps to train the policy for
        """
        obs = self.env.reset()
        r = []
        num = 0
        summ = 0
        t = 0
        for i in range(num_steps):
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            mu = self.actor(obs)
            m = MultivariateNormal(mu, covariance_matrix = self.policy_noise * torch.eye(2))
            action = m.sample()
            action = np.clip(action[0].numpy(), -1, 1)
            nobs, reward, done, _ = env.step(action)
            self.RB.add((obs.numpy().reshape(-1), action, reward, nobs, done))
            
            r.append(reward)
            num += 1
            summ += 1
            
            # update
            states, actions, rewards, next_states, dones = self.RB.sample(self.batch_size)
            next_actions = self.actor_target(next_states)
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-1, 1)
            Q_target1, Q_target2 = self.critic_target(next_states, next_actions)
            y = rewards + self.gamma * (1 - dones) * torch.min(Q_target1, Q_target2)
            Q1, Q2 = self.critic(states, actions)
            # update models
            self.critic_loss = F.mse_loss(Q1, y) + F.mse_loss(Q2, y)
            self.optimizer_critic.zero_grad()
            self.critic_loss.backward()
            self.optimizer_critic.step()
            
            if i % self.policy_delay == 0:
                self.actor_loss = - self.critic.Q(states, self.actor(states)).mean()
                self.optimizer_actor.zero_grad()
                self.actor_loss.backward()
                self.optimizer_actor.step()
                
                self.update_target_networks()
            
            
            if done: 
                avr = sum(r)
                self.avr.append(avr)
                print('Training %.2f%%, %d episodes, number of actions:%d, returns: %.3f'%(summ/num_steps * 100, t, num, avr))
                
                obs = self.env.reset()
                r = []
                t += 1
                num = 0
            else:
                obs = nobs

                          




