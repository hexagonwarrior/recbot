#!/bin/env python3


import argparse
import torch
from env import rec_env

global ENV
global MODEL

def init_rec():
    global ENV
    global MODEL

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        MODEL = torch.load("trained_models/recbot")
    else:
        MODEL = torch.load("trained_models/recbot", map_location=lambda storage, loc: storage)
    MODEL.eval()
    ENV = rec_env()
    ENV.reset()
    if torch.cuda.is_available():
        MODEL.cuda()
   

def get_rec_result():
    global ENV
    global MODEL

    actions, next_states, rewards = ENV.get_next_states()

    actions = torch.FloatTensor(actions)
    next_states = torch.FloatTensor(next_states)
    model_input = torch.cat((next_states, actions), 1) # 拼接state与action

    if torch.cuda.is_available():
        model_input = model_input.cuda()
    predictions = MODEL(model_input)[:, 0]
    index = torch.argmax(predictions).item()
    action = actions[index]
    next_state = next_states[index, :]
    ENV.step(next_state)
    
    return action.tolist()

if __name__ == '__main__':
    init_rec()
    print(get_rec_result())
