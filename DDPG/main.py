import gym
import torch
from utils import tic, toc, evaluate_policy
import matplotlib.pyplot as plt
from DDPG import DDPG
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed')
parser.add_argument('--num-steps', type=int, default=100000, metavar='N',
                    help='number of steps')
parser.add_argument('--actor_lr', type=float, default=1e-3, metavar='N',
                    help='learning rate for actor')
parser.add_argument('--critic_lr', type=float, default=1e-4, metavar='N',
                    help='learning rate for critic')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='batch size (default: 100)')

args = parser.parse_args

env = gym.make(args.env_name) 

env.seed(args.seed)
torch.manual_seed(args.seed)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

ddpg_object = DDPG(
    env,
    num_inputs,
    num_actions,
    args.critic_lr,
    args.actor_lr, 
    args.gamma,
    args.batch_size,
)
# Train the policy
print(f'Training with seed: {args.seed}')
a = tic()
ddpg_object.train(args.num_steps)
b = toc(a)
print(f'training time: {b}')

plt.plot(ddpg_object.avr)
plt.title(f'returns after every update when seed = {args.seed}')

## Evaluate the final policy
evaluate_policy(env, ddpg_object.actor)