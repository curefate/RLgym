import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class vPPO(nn.Module):
    def __init__(self, dim_actions):
        super().__init__()
        self.vision = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 22 * 16, 512)),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, dim_actions), std=0.01),
            nn.Softmax()
        )
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, state):
        return self.critic(self.vision(state))

    def select_action(self, state, action=None):
        logits = self.actor(self.vision(state))
        probs = torch.distributions.Categorical(logits)
        if action is None:
            action = probs.sample().item()
        return action, probs.log_prob(action), probs.entropy()

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "vision": self.vision.state_dict()
            },
            path)

    def load(self, path):
        print("load model:", path)
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.vision.load_state_dict(ckpt["vision"])


class vPPOtrainer():
    def __init__(self, agent:vPPO, args):
        self.agent = agent
        self.optimizer = torch.optim.Adam(self.agent.parameters(), args.lr)

    def optimize(self, transition_dict, args):
        actions_dict = torch.tensor(transition_dict['actions']).to(self.device).view(-1, 1)
        states_dict = torch.tensor([item.cpu().detach().numpy() for item in transition_dict['states']],
                              dtype=torch.float).to(self.device)
        rewards_dict = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        values_dict = torch.tensor(transition_dict['values'], dtype=torch.float).to(self.device).view(-1, 1)
        advantages_dict = torch.tensor(transition_dict['advantages'], dtype=torch.float).to(self.device).view(-1, 1)
        log_probs_dict = torch.tensor(transition_dict['log_probs'], dtype=torch.float).to(self.device).view(-1, 1)

        sep_counts = np.arange(args.batch_size)
        for epoch in range(args.epochs):
            np.random.shuffle(sep_counts)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                counter = sep_counts[start:end]

                _, log_probs, entropy = self.agent.select_action(states_dict[counter], actions_dict[counter])
                value = self.agent.get_value(states_dict[counter]).view(-1)
                ratio = torch.exp(log_probs - log_probs_dict[counter])
                advantages = advantages_dict[counter]
                if args.norm_adv:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss
                eps = args.clip_coef
                # 近端策略优化裁剪目标函数公式的左侧项
                loss_policy1 = -advantages * ratio
                # 公式的右侧项，裁剪ratio
                loss_policy2 = -advantages * torch.clamp(ratio, 1-eps, 1+eps)
                loss_policy = torch.max(loss_policy1, loss_policy2).mean()

                # Value loss
                if args.clip_vloss:
                    loss_value_unclipped = (value - rewards_dict[counter]) ** 2
                    loss_value_clipped = values_dict[counter] + torch.clamp(
                        value - values_dict[counter],
                        -eps, eps
                    )
                    loss_value_clipped = (loss_value_clipped - rewards_dict[counter]) ** 2
                    loss_value = .5 * torch.max(loss_value_unclipped, loss_value_clipped).mean()
                else:
                    loss_value = .5 * ((value - rewards_dict[counter]) ** 2).mean() # MSE loss

                # Entropy
                loss_entropy = entropy.mean()

                # Loss
                loss = loss_policy + args.lossv_coef * loss_value + args.losse_coef * loss_entropy

                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.agent.parameters(), args.max_grad_norm)
                self.optimizer.step()

        return loss

    def train(self, env):


if __name__ == '__main__':
    # sample = torch.randn(4, 210, 160)
    # model = vPPO(3)
    # out = model(sample)
    # print(out.shape)

    sample = torch.ones(3,3)
    obs = torch.zeros((4, 2) + sample.shape)
    print(obs.shape)
    obs[1] = sample
    obs[3] = sample