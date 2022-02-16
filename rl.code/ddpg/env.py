#!/usr/bin/env python3

import random
import torch
import numpy as np

CLASS_LEN = 8 # NEWS的种类数，共8种，
HISTORY_LEN = CLASS_LEN # 点击历史的计数
SEQ_LEN = 24 # 点击序列长度
STATE_LEN = HISTORY_LEN + SEQ_LEN # env的状态长度
ACTION_LEN = 8 # 每个动作包含8个推荐的NEWS
INPUT_LEN = STATE_LEN + STATE_LEN # modeL的input长度
ACTION_NUM = 8 # 每次step时的动作数
SAMPLE_PRECISION = 8 # 采样精度

class rec_env:
    state = [0 for i in range(STATE_LEN)] # 环境状态
    bot = [0 for i in range(CLASS_LEN)] # 模拟用的bot的点击分布

    def __init__(self):
        self.reset()

    def reset(self):
        self.state[0:HISTORY_LEN] = [0 for i in range(HISTORY_LEN)] 
        self.state[HISTORY_LEN:STATE_LEN] = [0 for i in range(SEQ_LEN)]
        # self.bot = [random.random() for i in range(CLASS_LEN)] # 生成8个随机数做为用户点击概率
        self.bot = [0.1, 0.1, 0.9, 0.1, 0.1, 0.4, 0.6, 0.5]
        return np.array(self.state)

    # 获取下一批状态，由随机的分布来采样8个动作
    def step(self, action):
        state = self.state[:] # 复制当前状态
        next_state, reward = self.emulate(action, state) # 模拟点击
        self.state[:]= next_state[:]
            
        return self.state, reward, None, None # 后两个done和extra只是占位，不被使用

    # 用户点击，用来与推荐系统交互
    def user_click(self, action):
        print("ACTION = ", action)
        click = []
        for c in action:
            p = self.bot[c - 1]
            if random.random() < p:
                click.append(c)

        print("CLICK = ", click)
        return click

    # 在对应的状态下，进行下一个动作
    def emulate(self, action, state):
        click = self.user_click(action[0].tolist())
        for i in click:
            state[i - 1] += 1
        count = len(click)
        state[HISTORY_LEN:STATE_LEN] = state[count + HISTORY_LEN:] + click[:] # FIXME
        reward = len(click) # 把点击数做为reward
        return state, reward
