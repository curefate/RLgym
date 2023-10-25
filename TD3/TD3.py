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

    def sample_tensor(self, batch_size, device):
        batch = random.sample(self.memory, batch_size)
        [states, actions, rewards, next_stages, dones] = zip(*batch)
        state_batch = torch.tensor(np.array(states)).to(device)
        action_batch = torch.tensor(np.array(actions)).to(device)
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
        self.noise_clip = args.noise_clip
        self.delay_times = args.delay_times
        self.batch_size = args.batch_size
        self.device = args.device
        self.model = model
        self.logs_path = args.logs_path
        self.save_path = args.save_path

        self.update_times = 0
        self.memory = Memory()

    def optimize(self):
        if self.memory.length() < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample_tensor(self.batch_size, self.device)

        with torch.no_grad():
            next_actions = self.model.target_actor(next_states)
            action_noise = (torch.randn(actions.shape) * .1).to(self.device)
            # smooth noise
            action_noise = torch.clamp(action_noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + action_noise, -1, 1)

            q1 = self.model.target_critic1(next_states, next_actions).view(-1)
            q2 = self.model.target_critic2(next_states, next_actions).view(-1)
            critic_value = torch.min(q1, q2)
            # TD predict
            predict_target_value = rewards + self.gamma * critic_value

        # all way predict
        q1 = self.model.target_critic1(states, actions).view(-1)
        q2 = self.model.target_critic2(states, actions).view(-1)

        # optimize critics
        critic1_loss = nn.functional.mse_loss(q1.float(), predict_target_value.detach().float())
        critic2_loss = nn.functional.mse_loss(q2.float(), predict_target_value.detach().float())
        critic_loss = (critic1_loss + critic2_loss).float()
        self.model.critic1.optimizer.zero_grad()
        self.model.critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.model.critic1.optimizer.step()
        self.model.critic2.optimizer.step()

        self.update_times += 1
        if self.update_times % self.delay_times != 0:
            return

        # optimize actor
        next_actions = self.model.actor(states)
        q1 = self.model.critic1(states, next_actions)
        q2 = self.model.critic2(states, next_actions)
        q = torch.min(q1, q2)
        actor_loss = -torch.mean(q)
        self.model.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.model.actor.optimizer.step()

        # model -----> ema
        self.model.accumulate(self.tau)

    def train(self, env, start_iter=0, end_iter=10001):
        logs = SummaryWriter(self.logs_path)
        pbar = range(end_iter)
        pbar = tqdm(pbar, initial=start_iter, dynamic_ncols=True, smoothing=.01)

        for idx in pbar:
            itercheck = idx + start_iter
            if itercheck > end_iter:
                print("Done!")
                break

            # reset state
            state, info = env.reset()
            done = False

            # start play
            while not done:
                # select action
                action = self.model.select_action(state, train=True)
                action = scale_action(action, low=env.action_space.low, high=env.action_space.high)

                # step
                next_state, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    done = True

                # storage memory
                self.memory.storage(state, action, reward, next_state, done)

                # next iter
                state = next_state

                # optimize
                self.optimize()

            # test
            if idx % 100 == 0:
                for i in range(100):
                    total_reward = 0
                    test_state, test_info = env.reset()
                    test_done = False
                    while not test_done:
                        test_action = self.model.select_action(test_state, train=False)
                        test_action = scale_action(test_action, low=env.action_space.low, high=env.action_space.high)
                        test_state, test_reward, test_terminated, test_truncated, test_info = env.step(test_action)
                        total_reward += test_reward
                        if test_terminated or test_truncated:
                            done = True
                            break
                avg_reward = total_reward / 100
                logs.add_scalar(
                    tag="Test Total Reward",
                    scalar_value=total_reward,
                    global_step=idx
                )
                logs.add_scalar(
                    tag="Test Average Reward",
                    scalar_value=avg_reward,
                    global_step=idx
                )

            # save
            if idx % 1000 == 0:
                self.model.save(self.save_path + f"/{str(idx).zfill(6)}.pt")

        print("Done!")


def scale_action(action, low, high):
    action = np.clip(action, -1, 1)
    weight = (high - low) / 2
    bias = (high + low) / 2
    action_ = action * weight + bias

    return action_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default='', help="path of model"
    )
    parser.add_argument(
        "--save_path", type=str, default='LunarLanderContinuous-v2/checkpoint', help="path to save"
    )
    parser.add_argument(
        "--logs_path", type=str, default='LunarLanderContinuous-v2/logs', help="path to logs"
    )
    parser.add_argument(
        "--end_iter", type=int, default=10001, help="end_iter"
    )
    parser.add_argument(
        "--start_iter", type=int, default=0, help="start_iter"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument(
        "--gamma", type=float, default=.99, help="discount constant"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="batch size"
    )
    parser.add_argument(
        "--tau", type=float, default=.005, help="accumulate rate"
    )
    parser.add_argument(
        "--noise_scale", type=float, default=.1, help="noise scale"
    )
    parser.add_argument(
        "--noise_clip", type=float, default=.5, help="noise clip"
    )
    parser.add_argument(
        "--delay_times", type=int, default=2, help="actor update once time when critics update delay times"
    )
    args = parser.parse_args()
    args.device = 'cpu'
    if torch.cuda.is_available():
        args.device = 'cuda'

    env = gym.make('LunarLanderContinuous-v2', render_mode=None)
    model = TD3(env.observation_space.shape[0], env.action_space.shape[0], args.device)
    if args.path != '':
        model.load(args.path)

    trainer = TD3_trainer(model, args)
    trainer.train(env, args.start_iter, args.end_iter)
    env.close()
