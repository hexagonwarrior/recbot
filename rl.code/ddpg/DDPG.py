from base_model import *
from copy import deepcopy
from env import rec_env


class DDPG:

    def __init__(self, state_dim, hidden_size, action_dim, env:rec_env, episodes, batch_size, buffer_size, lr=0.01, gamma=0.9, tau=0.02):
        self.env = env
        self.episodes = episodes
        self.replay_buffer = Replay_Buffer(buffer_size, batch_size)
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = Actor(state_dim, hidden_size, action_dim)
        self.actor_target = Actor(state_dim, hidden_size, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, hidden_size, action_dim)
        self.critic_target = Critic(state_dim, hidden_size, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.optim_for_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optim_for_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.tau = tau

    def soft_update(self, network: nn.Module, target_network: nn.Module):
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

    def choose_action(self, state):
        state_torch = torch.FloatTensor(state)
        action = self.actor(state_torch.unsqueeze(0))
        return action
    
    def learn(self):
        counters = []
        observation = self.env.reset()
        for episode in range(self.episodes):
            counter = 0
            done = 1
            while done > 0:
                counter += 1
                action = self.choose_action(observation)
                print('ACTION = ', action)
                observation_, reward, _, _ = self.env.step(action)
                self.replay_buffer.push(deepcopy(observation), action[0], reward, deepcopy(observation_), done)
                observation = observation_
                if not self.replay_buffer.is_full():
                    continue
                train_s, train_a, train_r, train_s_, train_done = self.replay_buffer.sample()
                train_s = torch.Tensor(train_s).view(-1, self.state_dim)
                # train_a = torch.Tensor(train_a).view(-1, self.action_dim)
                train_a = torch.stack(train_a, 0)
                train_r = torch.Tensor(train_r).view(-1, 1)
                train_s_ = torch.Tensor(train_s_).view(-1, self.state_dim)
                train_done = torch.Tensor(train_done).view(-1, 1)
                q_pre = self.critic(train_s, train_a)

                a_ = torch.stack(self.actor_target(train_s_), 0)
                q_target = self.critic_target(train_s_, a_)
                q_target = train_r + self.gamma*q_target*(1-train_done)

                loss_for_critic = ((q_pre-q_target)**2).mean()
                self.optim_for_critic.zero_grad()
                loss_for_critic.backward()
                self.optim_for_critic.step()

                a_ = torch.stack(self.actor(train_s), 0)
                loss_for_actor = -self.critic(train_s, a_).mean()
                self.optim_for_actor.zero_grad()
                loss_for_actor.backward()
                self.optim_for_actor.step()
                self.soft_update(self.actor, self.actor_target)
                self.soft_update(self.critic, self.critic_target)
                done -= 1
            counters.append(counter)
            print("episode {} end, stay for {} steps".format(episode+1, counter))
            if len(counters)>5 and counters[-5]==200 and counters[-4]==200 and counters[-3]==200 and counters[-2]==200 and counters[-1]==200:
                return counters
        return counters
