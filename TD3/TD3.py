import math
import os
import random
import gymnasium as gym
import torch
from torch import nn
from tqdm import tqdm
import argparse
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim=128, fc2_dim=256, lr=1e-4):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr)

    def forward(self, state):
        temp = nn.functional.relu(self.ln1(self.fc1(state)))
        temp = nn.functional.relu(self.ln2(self.fc2(temp)))
        ret = nn.functional.tanh(self.fc3(temp))
        return ret


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim=128, fc2_dim=256, lr=1e-4):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr)

    def forward(self, state, action):
        temp = torch.cat([state, action], dim=-1)
        temp = nn.functional.relu(self.ln1(self.fc1(temp)))
        temp = nn.functional.relu(self.ln2(self.fc2(temp)))
        ret = self.fc3(temp)
        return ret


class TD3:
    def __init__(self, state_dim, action_dim, device, actor_fc1_dim=128, actor_fc2_dim=256, actor_lr=1e-4,
                 critic_fc1_dim=128, critic_fc2_dim=256, critic_lr=1e-4):
        self.device = device
        self.actor = Actor(state_dim, action_dim, actor_fc1_dim, actor_fc2_dim, actor_lr).to(device)
        self.target_actor = Actor(state_dim, action_dim, actor_fc1_dim, actor_fc2_dim, actor_lr).to(device)
        self.critic1 = Critic(state_dim, action_dim, critic_fc1_dim, critic_fc2_dim, critic_lr).to(device)
        self.target_critic1 = Critic(state_dim, action_dim, critic_fc1_dim, critic_fc2_dim, critic_lr).to(device)
        self.critic2 = Critic(state_dim, action_dim, critic_fc1_dim, critic_fc2_dim, critic_lr).to(device)
        self.target_critic2 = Critic(state_dim, action_dim, critic_fc1_dim, critic_fc2_dim, critic_lr).to(device)

    def select_action(self, state, train=False, noise_scale=.1):
        self.actor.eval()
        state = torch.tensor(state).view(-1, len(state)).to(self.device)
        action = self.actor(state)

        if train:
            noise = torch.randn(action.shape).to(self.device) * noise_scale
            action = torch.clamp(action + noise, -1, 1)

        self.actor.train()
        return action.max(1)[1].item()

    def accumulate(self, tau):
        for actor_params, target_actor_params in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)
        for critic1_params, target_critic1_params in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_critic1_params.data.copy_(tau * critic1_params + (1 - tau) * target_critic1_params)
        for critic2_params, target_critic2_params in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_critic2_params.data.copy_(tau * critic2_params + (1 - tau) * target_critic2_params)

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target_critic2": self.target_critic2.state_dict()
            },
            path)

    def load(self, path):
        print("load model:", path)
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.actor.load_state_dict(ckpt["actor"])
        self.target_actor.load_state_dict(ckpt["target_actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.target_critic1.load_state_dict(ckpt["target_critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.target_critic2.load_state_dict(ckpt["target_critic2"])


class Memory:
    def __init__(self, max_size=100000):
        self.memory = deque(maxlen=max_size)

    def storage(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size, device):
        batch = random.sample(self.memory, batch_size)
        [states, actions, rewards, next_stages, dones] = zip(*batch)
        state_batch = torch.tensor(np.array(states)).to(device)
        action_batch = torch.tensor(actions).to(device)
        reward_batch = torch.tensor(rewards).to(device)
        next_state_batch = torch.tensor(np.array(next_stages)).to(device)
        dones_batch = torch.tensor(dones).to(device)
        return state_batch, action_batch, reward_batch, next_state_batch, dones_batch

    def length(self):
        return len(self.memory)


class TD3_trainer:
    def __init__(self, model: TD3, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.noise_scale = args.noise_scale
        self.delay_times = args.delay_times
        self.batch_size = args.batch_size
        self.device = args.device

        self.update_times = 0
        self.memory = Memory()

    def optimize(self):
        if self.memory.length() < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device)
        