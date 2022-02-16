#!/usr/bin/env python3

import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn

from deep_q_network import DeepQNetwork
from env import rec_env
from collections import deque


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network for News recommendation""")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=10)
    # parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    env = rec_env()
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    while epoch < opt.num_epochs:
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon

        actions, next_states, rewards = env.get_next_states()
        print("NEXT_ACTIONS = ", actions)
        print("NEXT_STATES = ", next_states)

        actions = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)
        model_input = torch.cat((next_states, actions), 1) # 拼接state与action

        # print("INPUT = ", model_input)
        if torch.cuda.is_available():
            model_inputs = model_inputs_.cuda()
        model.eval()

        with torch.no_grad():
            predictions = model(model_input)[:, 0]
        model.train()
        
        # 选择动作
        if random_action:
            index = randint(0, len(next_states) - 1)
        else:
            index = torch.argmax(predictions).item()

        # 行动
        action = actions[index]
        reward = rewards[index]
        next_state = next_states[index, :]
        env.step(next_state)

        replay_memory.append([torch.FloatTensor(state), action, reward, next_state]) # <s_t, a, r, s_t+1>
        state = next_state

        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        epoch += 1 # 经验池每涨1/10，epoch增1

        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch = zip(*batch)

        state_batch = torch.stack(tuple(state for state in state_batch))
        action_batch = torch.stack(tuple(action for action in action_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = model(torch.cat((state_batch, action_batch), 1))
        model.eval()
        with torch.no_grad():
            best_action = env.get_batch_action(next_state_batch.tolist())
            dqn_input = torch.cat((next_state_batch, torch.FloatTensor(best_action)), 1)

            if torch.cuda.is_available():
                dqn_input = dqn_input.cuda()

            print("DQN_INPUT = ", dqn_input)
            next_prediction_batch = model(dqn_input)
        model.train()

        y_batch = torch.cat(
            tuple(reward + opt.gamma * prediction for reward, prediction in
                  zip(reward_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}".format(
            epoch,
            opt.num_epochs,
            ))

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/recbot_{}".format(opt.saved_path, epoch))

    torch.save(model, "{}/recbot".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
