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


class DQN_Network(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_actions)

    def forward(self, s):
        ret = nn.functional.relu(self.fc1(s))
        ret = nn.functional.relu(self.fc2(ret))
        ret = self.fc3(ret)
        return ret


class DQN_trainer():
    def __init__(self, args):
        super().__init__()

        # hyper parameters
        self.lr = args.lr
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.device = args.device
        self.save_path = args.save_path
        self.logs_path = args.logs_path

    def select_action(self, model, state, eps):
        if random.random() <= eps:
            return random.randint(0, 1)
        else:
            ret = model(torch.tensor(state).view(-1, 4).to(self.device))
            return ret.max(1)[1].item()

    def optimize(self, model, optimizer, memory):
        if len(memory) < self.batch_size:
            return
        batch = random.sample(memory, self.batch_size)
        [states, actions, rewards, next_stages, dones] = zip(*batch)
        state_batch = torch.tensor(np.array(states)).to(self.device)
        action_batch = torch.tensor(actions).to(self.device)
        reward_batch = torch.tensor(rewards).to(self.device)
        next_state_batch = torch.tensor(np.array(next_stages)).to(self.device)

        # 预测全部Q值(qt)
        real_state_action_value = model(state_batch).gather(1, action_batch.view(-1, 1))

        # 计算TD target(yt)
        next_state_value = model(next_state_batch).max(1)[0]
        for i in range(self.batch_size):
            if dones[i]:
                next_state_value[i] = 0
        predict_state_action_value = (reward_batch + next_state_value * self.gamma).view(-1, 1)

        # 计算TD loss
        loss = nn.functional.mse_loss(real_state_action_value, predict_state_action_value)

        # 优化
        optimizer.zero_grad()
        loss.backward()
        # TODO 梯度裁剪
        optimizer.step()
        return loss.item()

    def train(self, env, model, start_iter=0, end_iter=10000):
        logs = SummaryWriter(self.logs_path)
        pbar = range(end_iter)
        pbar = tqdm(pbar, initial=start_iter, dynamic_ncols=True, smoothing=.01)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        memory = deque(maxlen=10000)

        for idx in pbar:
            itercheck = idx + start_iter
            if itercheck > end_iter:
                print("Done!")
                break

            # reset state
            state, info = env.reset()

            # play once time
            for i in range(99999):  # max step
                # 选择动作
                eps = self.eps_end + (self.eps_start - self.eps_end) / math.exp(-1. * idx / self.eps_decay)
                action = self.select_action(model, state, eps)

                # step
                next_state, reward, terminated, truncated, info = env.step(action)

                # 记录
                done = False
                if terminated or truncated:
                    done = True
                memory.append([state, action, reward, next_state, done])

                # 下一轮
                state = next_state

                # optimize
                if len(memory) >= self.batch_size:
                    loss = self.optimize(model, optimizer, memory)
                    logs.add_scalar(
                        tag="Loss",
                        scalar_value=loss,
                        global_step=idx
                    )

                if done:
                    break

            if idx % 100 == 0:
                for i in range(30):
                    total_reward = 0
                    test_state, test_info = env.reset()
                    for j in range(99999):
                        test_action = self.select_action(model, test_state, 0)
                        test_state, test_reward, test_terminated, test_truncated, test_info = env.step(test_action)
                        total_reward += test_reward
                        if test_terminated or test_truncated:
                            break
                logs.add_scalar(
                    tag="Test Total Reward",
                    scalar_value=total_reward,
                    global_step=idx
                )
            if idx % 1000 == 0:
                torch.save({"model": model.state_dict()},
                           (self.save_path + f"/{str(idx).zfill(6)}.pt"))
        print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default='', help="path of model"
    )
    parser.add_argument(
        "--save_path", type=str, default='DQN/Cartpole/checkpoint', help="path to save"
    )
    parser.add_argument(
        "--logs_path", type=str, default='DQN/Cartpole/logs', help="path to logs"
    )
    parser.add_argument(
        "--end_iter", type=int, default=10000, help="end_iter"
    )
    parser.add_argument(
        "--start_iter", type=int, default=0, help="start_iter"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument(
        "--eps_start", type=float, default=.5, help="initial value of epsilon"
    )
    parser.add_argument(
        "--eps_end", type=float, default=.01, help="final value of epsilon"
    )
    parser.add_argument(
        "--eps_decay", type=float, default=1000., help="decay speed"
    )
    parser.add_argument(
        "--gamma", type=float, default=.99, help="discount constant"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="batch size"
    )
    args = parser.parse_args()
    args.device = 'cpu'
    if torch.cuda.is_available():
        args.device = 'cuda'

    env = gym.make('CartPole-v1', render_mode=None)
    model = DQN_Network(4, 2).to(args.device)
    if args.path != '':
        print("load model: {}".format(args.path))
        ckpt = torch.load(args.path, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
            args.end_iter = args.start_iter + 10000
        except ValueError:
            pass
        model.load_state_dict(ckpt["model"])

    trainer = DQN_trainer(args)
    trainer.train(env, model, args.start_iter, args.end_iter)
    env.close()
