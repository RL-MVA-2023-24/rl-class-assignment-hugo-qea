from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import numpy as np
import torch.nn as nn
import os
from copy import deepcopy
import torch.nn.functional as F

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_PATH = "agentDQNL1_v3.pth"

LOAD_PATH = os.path.join(os.path.dirname(__file__), "agentDQNL1_v3.pth")


n_actions = env.action_space.n
n_neurons = 256
n_states = env.observation_space.shape[0]
depth = 10



class DQN(nn.Module):
    def __init__(self, n_states, n_actions, n_neurons, depth, device):
        super(DQN, self).__init__()
        self.device = device
        self.layers = nn.ModuleList([nn.Linear(n_states, n_neurons)])
        for _ in range(depth):
            self.layers.append(nn.Linear(n_neurons, n_neurons))
        self.layers.append(nn.Linear(n_neurons, n_actions))
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)
    
model = DQN(n_states, n_actions, n_neurons, depth, device).to(device)

config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.90,
          'buffer_size': 10000000,
          'epsilon_min': 0.02,
          'epsilon_max': 0.02,
          'epsilon_decay_period': 10000,
          'epsilon_delay_decay': 50,
          'batch_size': 512,
          'gradient_steps': 10,
          'update_target_strategy': 'ema', # or 'ema'
          'update_target_freq': 50,
          'update_target_tau': 0.005,
          'criterion': nn.SmoothL1Loss(),
          'device': device,
          'optimizer': torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.95, eps=0.01)}


def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

    


class ProjectAgent:
    def __init__(self, config=config, model=model):
        self.device = config['device']
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
                
            if episode % 25 == 0:
                self.save(SAVE_PATH+"_tmp"+str(episode))
        return episode_return
    
    def act(self, observation, use_random=False):
        with torch.no_grad():
            if use_random:
                return env.action_space.sample()
            else:
                return greedy_action(self.model, observation)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load(LOAD_PATH, map_location='cpu'))



if __name__ == "__main__":
    # Train the agent
    agent = ProjectAgent(config, model)
    agent.load()
    agent.train(env, 200)
    agent.save(SAVE_PATH)
    """
    config = {'gamma': .99,
          'buffer_size': 1e6,
          'learning_rate': 3e-4,
          'batch_size': 256,
          'tau': 0.005,
          'delay_learning': 1e4,
          'exploration_noise': .1,
          'action_noise_scale': 0.2,
          'action_noise_clip"': 0.5,
          'policy_update_freq': 2,
         }
    Qfunction1 = QNetwork(env).to(device)
    Qfunction2 = QNetwork(env).to(device)
    policy = policyNetwork(env).to(device)
    agent = td3_agent(config, Qfunction1, Qfunction2, policy)
    agent.train(env, 200)"""