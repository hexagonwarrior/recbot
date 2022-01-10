#!/usr/bin/env python3

import random
import torch

CLASS_LEN = 8 # NEWS的种类数，共8种，
HISTORY_LEN = CLASS_LEN # 点击历史的计数
SEQ_LEN = 16 # 点击序列长度
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
        self.bot = [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.8, 0.5]
        return self.state

    # 根据点击历史来计算分布
    # for example, history = [97, 82, 14, 59, 52, 56, 10, 0] 8类NEWS的点击数
    def get_distributation(self, history):
        # 计算分布
        distribution = [0 for _ in range(CLASS_LEN)]
        total = sum(history)
        s = 0
        for k in range(len(history)):
            s += history[k] / total
            distribution[k] = s
        print("NEXT STATES DISTRIBUTION = ", distribution)
        return distribution

    # 根据分布来采样动作
    def get_action(self, distribution):
        # 根据分布采样
        action = [0 for _ in range(ACTION_LEN)]
        for i in range(SAMPLE_PRECISION): # 采样精度
            r = random.random()
            for j in range(len(distribution)):
                if distribution[j] > r:
                    action[i] = j + 1
                    break
            # print(r, j)
        print("ACTION = ", action)
        return action

    # 获取下一批状态，由随机的分布来采样8个动作
    def get_next_states(self):
        actions = []
        states = []
        rewards = []
        for i in range(ACTION_NUM): # 取8个推荐做为动作
            history = [random.randint(0, 100) for _ in range(CLASS_LEN)] # 随机取一个点样本
            distribution = self.get_distributation(history) # 计算分布
            action = self.get_action(distribution) # 生成8种不同分布的随机动作
            state = self.state[:] # 复制当前状态
            print("GET_NEXT_STATES = ", state)
            next_state, reward = self.emulate(action, state) # 模拟下一个动作，产生下一个状态
            actions.append(action)
            states.append(next_state)
            rewards.append(reward)
            
        return actions, states, rewards # 反回8个动作模拟和产生的状态字典

    # 用户点击，用来与推荐系统交互
    def user_click(self, action):
        print("CLICK ACTION = ", action)
        click = []
        for c in action:
            p = self.bot[c - 1]
            if random.random() < p:
                click.append(c)

        print("CLICK = ", click)
        return click

    # 在对应的状态下，进行下一个动作
    def emulate(self, action, state):
        click = self.user_click(action)
        for i in click:
            state[i - 1] += 1
        count = len(click)
        state[HISTORY_LEN:STATE_LEN] = state[count + HISTORY_LEN:] + click[:] # FIXME
        reward = len(click) # 把点击数做为reward
        return state, reward

    def step(self, state):
        self.state[:]= state.tolist()[:]
        
    def get_batch_action(self, next_state_batch):
        actions = []
        for i in range(len(next_state_batch)):
            actions.append(self.get_best_action(next_state_batch[i][:HISTORY_LEN]))
        return actions
    
    def get_best_action(self, history):
        print("GET_BEST_ACTION = ", history)
        distribution = self.get_distributation(history) # 计算当前分布
        action = self.get_action(distribution) 
        return action
 
