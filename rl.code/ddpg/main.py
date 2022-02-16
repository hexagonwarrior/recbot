#!/bin/env python3
from DDPG import *
import matplotlib.pyplot as plt
from env import rec_env

if __name__ == '__main__':
    env = rec_env()

    # state_dim, hidden_size, action_dim
    # env:rec_env, episodes, batch_size, buffer_size, 
    # lr=0.01, gamma=0.9, tau=0.02
    ddpg = DDPG(
        32, # state_dim, 
        64, # hidden_size,
        8, # action_dim,
        env, # env:rec_env,
        200, # episodes,
        32, # batch_size,
        512, # buffer_size,
        0.1 # lr=0.01
    ) 
    steps = ddpg.learn()
    # episodes = list(range(1, len(steps)+1))
    # plt.plot(episodes, steps)
    # plt.xlabel('episodes')
    # plt.ylabel('steps')
    # plt.title('steps-episodes')
    # plt.legend()
    # plt.savefig('steps_episodes.svg')
