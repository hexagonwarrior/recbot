#!/usr/bin/env python3

import argparse
import torch
from env import rec_env


def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to RecBot""")

    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        model = torch.load("{}/recbot".format(opt.saved_path))
    else:
        model = torch.load("{}/recbot".format(opt.saved_path), map_location=lambda storage, loc: storage)
    model.eval()
    env = rec_env()
    env.reset()
    if torch.cuda.is_available():
        model.cuda()
    count = 10
    while count > 0:
        actions, next_states, rewards = env.get_next_states()

        actions = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)
        model_input = torch.cat((next_states, actions), 1) # 拼接state与action

        if torch.cuda.is_available():
            model_input = model_input.cuda()
        predictions = model(model_input)[:, 0]
        index = torch.argmax(predictions).item()
        action = actions[index]
        next_state = next_states[index, :]
        env.step(next_state)

        count -= 1
        
if __name__ == "__main__":
    opt = get_args()
    test(opt)
