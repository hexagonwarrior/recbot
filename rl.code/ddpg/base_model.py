import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):

    def __init__(self, state_dim, hidden_size, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.seq = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.action_dim * 2)
        )
        for layer in self.seq: # 对每一层参数进行初始化
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.01)
    
    def forward(self, state: torch.Tensor)->torch.Tensor:
        state = state.view(-1, self.state_dim)
        n = self.seq(state).tolist()
        actions = []
        for i in range(len(n)):
            u = torch.FloatTensor(n[i][:self.action_dim])
            s = torch.FloatTensor(np.abs(np.array(n[i][self.action_dim:self.action_dim*2])))
            action = torch.sigmoid(torch.normal(u, s))
            actions.append((action * 8 + 1).int())
        return actions

class Critic(nn.Module):

    def __init__(self, state_dim, hidden_size, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.seq = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )
        for layer in self.seq:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.01)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor)->torch.Tensor:
        state = state.view(-1, self.state_dim)
        action = action.view(-1, self.action_dim)
        print('state.shape = ', state.shape)
        print('action.shape = ', action.shape)
        x = torch.cat([state, action], dim=1)
        x = self.seq(x)
        return x


class Replay_Buffer:

    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.ptr = 0
        self.buffer = []
    
    def push(self, s, a, r, s_, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append((s, a, r, s_, done))
            self.ptr = (self.ptr+1)%self.max_size
        else:
            self.buffer[self.ptr] = (s, a, r, s_, done)
            self.ptr = (self.ptr+1)%self.max_size
    
    def sample(self):
        indexs = np.random.choice(range(self.max_size), self.batch_size)
        s = []
        a = []
        r = []
        s_ = []
        done = []
        for idx in indexs:
            s1, a1, r1, s_1, done1 = self.buffer[idx]
            s.append(s1)
            a.append(a1)
            r.append(r1)
            s_.append(s_1)
            done.append(done1)
        
        return s, a, r, s_, done
    
    def is_full(self):
        return len(self.buffer)==self.max_size
