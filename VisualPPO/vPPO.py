import torch
import gymnasium as gym
import torchvision
import win32con
import win32gui
from PIL import ImageGrab
from torch import nn


def get_window(window_name):
    handle = win32gui.FindWindow(0, window_name)
    # 获取窗口句柄
    if handle == 0:
        return None
    else:
        # 返回坐标值和handle
        return win32gui.GetWindowRect(handle), handle


def fetch_image():
    (x1, y1, x2, y2), handle = get_window("pygame window")
    x1 += 8
    x2 += -8
    y1 += 31
    y2 += -8
    win32gui.SendMessage(handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
    win32gui.SetForegroundWindow(handle)
    grab_image = ImageGrab.grab((x1, y1, x2, y2))
    return grab_image


class VisualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, (3, 3), padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channel, out_channel, (3, 3), padding=1, stride=(2, 2))
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        return out


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.res1 = VisualBlock(3, 4)
        self.res2 = VisualBlock(4, 6)
        self.res3 = VisualBlock(6, 8)
        self.fc1 = nn.Linear(30000, state_dim)
        self.fc2 = nn.Linear(state_dim, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.res3(self.res2(self.res1(state))).view(-1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        ret = nn.functional.softmax(self.fc3(x), dtype=torch.double)
        print(ret.shape)
        return ret


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.res1 = VisualBlock(3, 4)
        self.res2 = VisualBlock(4, 6)
        self.res3 = VisualBlock(6, 8)
        self.fc1 = nn.Linear(30000, state_dim)
        self.fc2 = nn.Linear(state_dim, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state):
        x = self.res3(self.res2(self.res1(state))).view(-1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        ret = self.fc3(x)
        return ret


class vPPO:
    def __init__(self, state_dim, action_dim, device, lr=3e-4):
        self.actor = PolicyNet(state_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)

        self.img2tensor = torchvision.transforms.ToTensor()
        self.device = device

    def select_action(self, state):
        state = state.to(self.device)
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


if __name__ == '__main__':
    env = gym.make('LunarLander-v2', render_mode="human")

    device = "cuda"
    model = vPPO(512, env.action_space.n, device)
    # model.load("CartPole-v1/checkpoint/002000.pt")
    for iters in range(10):
        done = False
        state, info = env.reset()
        total_reward = 0
        while not done:
            state = model.img2tensor(fetch_image())
            action = model.select_action(state)
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                done = True

        print("{}th test, total reward={}".format(iters, total_reward))

    # sample = torch.randn(3, 600, 400)
    # res1 = VisualBlock(3, 4)
    # res2 = VisualBlock(4, 6)
    # res3 = VisualBlock(6, 8)
    # out = res3(res2(res1(sample))).view(-1)
    # print(out.shape) # torch.Size([30000])
