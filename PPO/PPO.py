import argparse

import torch
import gymnasium as gym
from torch import nn
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, action_dim)

    def forward(self, state):
        x = nn.functional.relu(self.fc1(state))
        x = nn.functional.relu(self.fc2(x))
        ret = nn.functional.softmax(self.fc3(x), dim=1)
        return ret


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, state):
        x = nn.functional.relu(self.fc1(state))
        x = nn.functional.relu(self.fc2(x))
        ret = self.fc3(x)
        return ret


class PPO:
    def __init__(self, state_dim, action_dim, device, lr=3e-4):
        self.actor = PolicyNet(state_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)

        self.device = args.device

    def select_action(self, state):
        state = torch.tensor(state).view(-1, len(state)).to(self.device)
        probs = self.actor(state)
        action_list = torch.distributions.Categorical(probs)
        action = action_list.sample().item()
        return action

    def optimize(self, transition_dict, gamma, lmbda, eps, epochs):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1, 1)

        # calculate TD
        next_q_target = self.critic(next_states)
        td_target = rewards + gamma * next_q_target * (1 - dones)
        td_value = self.critic(states)
        td_delta = td_target - td_value

        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0
        advantage_list = []

        # calculate advantage function
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # GAE
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

        # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        avg_actor_loss = 0
        avg_critic_loss = 0

        for _ in range(epochs):
            # 每一轮更新一次策略网络预测的状态
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 新旧策略之间的比例
            ratio = torch.exp(log_probs - old_log_probs)
            # 近端策略优化裁剪目标函数公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage

            # 策略网络的损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            avg_actor_loss += actor_loss.item()
            # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
            critic_loss = torch.mean(torch.nn.functional.mse_loss(self.critic(states), td_target.detach()))
            avg_critic_loss += critic_loss.item()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return avg_actor_loss / epochs, avg_critic_loss / epochs

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path)

    def load(self, path):
        print("load model:", path)
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


def train(agent: PPO, env, args):
    logs = SummaryWriter(args.logs_path)
    pbar = range(args.end_iter)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=.01)

    for idx in pbar:
        itercheck = idx + args.start_iter
        if itercheck > args.end_iter:
            print("Done!")
            break

        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }

        # reset state
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = True

            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)

            # go next
            state = next_state

            # log reward
            total_reward += reward

        # optimize
        avg_actor_loss, avg_critic_loss = agent.optimize(transition_dict, args.gamma, args.lmbda, args.eps, args.epochs)
        logs.add_scalar("reward", total_reward, idx)
        logs.add_scalar("average actor loss", avg_actor_loss, idx)
        logs.add_scalar("average critic loss", avg_critic_loss, idx)

        # save
        if idx % 500 == 0:
            agent.save(args.save_path + f"/{str(idx).zfill(6)}.pt")

    print("done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default='', help="path of model"
    )
    parser.add_argument(
        "--save_path", type=str, default='MountainCar-v0/checkpoint', help="path to save"
    )
    parser.add_argument(
        "--logs_path", type=str, default='MountainCar-v0/logs', help="path to logs"
    )
    parser.add_argument(
        "--end_iter", type=int, default=3001, help="end_iter"
    )
    parser.add_argument(
        "--start_iter", type=int, default=0, help="start_iter"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="learning rate"
    )
    parser.add_argument(
        "--gamma", type=float, default=.99, help="discount constant"
    )
    parser.add_argument(
        "--lmbda", type=float, default=.95, help="scaler of GAE"
    )
    parser.add_argument(
        "--eps", type=float, default=.2, help="clip"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="how many times of one data trained"
    )
    args = parser.parse_args()
    args.device = 'cpu'
    if torch.cuda.is_available():
        args.device = 'cuda'

    env = gym.make('MountainCar-v0', render_mode=None)
    model = PPO(env.observation_space.shape[0], 2, args.device, args.lr)
    if args.path != '':
        model.load(args.path)

    train(model, env, args)
    env.close()
