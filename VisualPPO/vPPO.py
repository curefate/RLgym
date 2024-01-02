import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import argparse
import os

from tqdm import tqdm


def layer_init(layer, std=np.sqrt(2), bias_const=.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode=None)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        # env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class vPPO(nn.Module):
    def __init__(self, dim_actions):
        super().__init__()
        self.vision = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, (8, 8), stride=(4, 4))),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, (4, 4), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, (3, 3), stride=(1, 1))),
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

    def get_values(self, state, action=None):
        state = state / 255.
        logits = self.actor(self.vision(state))
        probs = torch.distributions.Categorical(logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(self.vision(state))

    def select_action(self, state):
        state = state / 255.
        logits = self.actor(self.vision(state))
        probs = torch.distributions.Categorical(logits)
        action = probs.sample()
        return action

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


def train(agent: vPPO, envs, args):
    # Setup
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Initialize
    storage_observations = torch.zeros((args.num_rollout_step, args.num_envs) + (args.num_skip_frame, 210, 160)).to(
        args.device)
    storage_actions = torch.zeros((args.num_rollout_step, args.num_envs) + envs.single_action_space.shape).to(
        args.device)
    storage_log_probs = torch.zeros((args.num_rollout_step, args.num_envs)).to(args.device)
    storage_rewards = torch.zeros((args.num_rollout_step, args.num_envs)).to(args.device)
    storage_dones = torch.zeros((args.num_rollout_step, args.num_envs)).to(args.device)
    storage_values = torch.zeros((args.num_rollout_step, args.num_envs)).to(args.device)

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    # Start
    next_obs, _ = envs.reset()
    # action = torch.zeros(args.num_envs)
    next_obs = torch.tensor(next_obs)[:, None, :, :].to(args.device)
    # skip frame
    # for i in range(args.num_skip_frame - 1):
    #     temp_obs, _, temp_terminated, temp_truncated, _ = envs.step(action.int().tolist())
    #     next_obs = torch.cat([next_obs, torch.tensor(temp_obs)[:, None, :, :].to(args.device)], dim=1)
    #     for d in range(len(temp_terminated)):
    #         if temp_terminated[d] or temp_truncated[d]:
    #             temp_terminated[d] = True
    #     next_done = torch.tensor(temp_terminated).to(args.device)
    next_done = torch.zeros(args.num_envs)

    global_step = 0  # for log
    roll_out_times = args.num_total_time_steps // args.batch_size
    pbar = tqdm(total=args.num_total_time_steps)
    for update in range(1, roll_out_times + 1):
        # Annealing learning rate
        if args.anneal_lr:
            frac = 1. - (update - 1.) / roll_out_times
            optimizer.param_groups[0]['lr'] = frac * args.lr

        # Rollout
        for step in range(args.num_rollout_step):
            pbar.update(1 * args.num_envs)
            global_step += 1 * args.num_envs
            storage_observations[step] = next_obs
            storage_dones[step] = next_done

            with torch.no_grad():
                action, log_probs, _, value = agent.get_values(next_obs)
            storage_actions[step] = action
            storage_log_probs[step] = log_probs
            storage_values[step] = value.flatten()

            next_obs, reward, terminated, truncated, _ = envs.step(action.int().tolist())
            for d in range(len(terminated)):
                if terminated[d] or truncated[d]:
                    terminated[d] = True
            done = terminated
            next_obs = torch.tensor(next_obs)[:, None, :, :].to(args.device)
            # skip frame
            # for i in range(args.num_skip_frame - 1):
            #     temp_obs, temp_reward, temp_terminated, temp_truncated, info = envs.step(action.int().tolist())
            #     next_obs = torch.cat([next_obs, torch.tensor(temp_obs)[:, None, :, :].to(args.device)], dim=1)
            #     reward += temp_reward
            #     for d in range(len(temp_terminated)):
            #         if temp_terminated[d] or temp_truncated[d]:
            #             temp_terminated[d] = True
            #     done = temp_terminated
            storage_rewards[step] = torch.tensor(reward).to(args.device).view(-1)
            # Go next
            next_done = torch.tensor(done).to(args.device)

        # todo

        # Learning: Calculate GAE
        with torch.no_grad():
            _, _, _, next_value = agent.get_values(next_obs)
            td_targets = storage_rewards + args.gamma * next_value.view(-1, args.num_envs) * (1 - storage_dones)
            td_values = storage_values
            td_delta = (td_targets - td_values).cpu().detach().numpy()
            advantage = 0
            storage_advantages = []
            for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
                # GAE
                advantage = args.gamma * args.lmbda * advantage + delta
                storage_advantages.append(advantage)
            storage_advantages.reverse()
            storage_advantages = torch.tensor(storage_advantages, dtype=torch.float).to(args.device)

        # Reshape storage
        storage_opt_observations = storage_observations.reshape((-1,) + (args.num_skip_frame, 210, 160))
        storage_opt_log_probs = storage_log_probs.reshape(-1)
        storage_opt_advantages = storage_advantages.reshape(-1)
        storage_opt_actions = storage_actions.reshape((-1,) + envs.single_action_space.shape)
        storage_opt_values = storage_values.reshape(-1)

        # Learning: Optimize
        sep_counts = np.arange(args.batch_size)
        for epoch in range(args.num_epochs):
            np.random.shuffle(sep_counts)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                idx = sep_counts[start:end]

                _, log_probs, entropy, value = agent.get_values(storage_opt_observations[idx],
                                                                storage_opt_actions[idx])
                # value = value.view(-1)
                ratio = torch.exp(log_probs - storage_opt_log_probs[idx])
                advantages = storage_opt_advantages[idx]
                if args.norm_adv:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss
                eps = args.clip_coef
                # 近端策略优化裁剪目标函数公式的左侧项
                loss_policy1 = -advantages * ratio
                # 公式的右侧项，裁剪ratio
                loss_policy2 = -advantages * torch.clamp(ratio, 1 - eps, 1 + eps)
                loss_policy = torch.max(loss_policy1, loss_policy2).mean()

                # Value loss
                if args.clip_vloss:
                    loss_value_unclipped = (value - (storage_opt_advantages[idx] + storage_opt_values[idx])) ** 2
                    loss_value_clipped = storage_opt_values[idx] + torch.clamp(
                        value - storage_opt_values[idx],
                        -eps, eps
                    )
                    loss_value_clipped = (loss_value_clipped - (
                            storage_opt_advantages[idx] + storage_opt_values[idx])) ** 2
                    loss_value = .5 * torch.max(loss_value_unclipped, loss_value_clipped).mean()
                else:
                    loss_value = .5 * ((value - (
                            storage_opt_advantages[idx] + storage_opt_values[idx])) ** 2).mean()  # MSE loss

                # Entropy
                loss_entropy = entropy.mean()

                # Loss
                loss = loss_policy + args.lossv_coef * loss_value + args.losse_coef * loss_entropy

                # optimize
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # log
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", loss_value.item(), global_step)
        writer.add_scalar("losses/policy_loss", loss_policy.item(), global_step)
        writer.add_scalar("losses/entropy", loss_entropy.item(), global_step)

        # save
        if global_step % (args.num_rollout_step * args.num_envs * 100) == 0:
            path = "ckpt/"
            if not os.path.exists(path):
                os.mkdir(path)
            agent.save(path + run_name + f"_{str(global_step + args.already_trained_times).zfill(6)}.pt")

    writer.close()
    return


if __name__ == '__main__':
    # sample = torch.randn(4, 210, 160)
    # model = vPPO(3)
    # out = model(sample)
    # print(out.shape)

    envs = gym.vector.SyncVectorEnv([
        lambda: gym.make("ALE/Assault-v5", obs_type="grayscale") for i in range(8)
    ])
    obs, info = envs.reset()
    obs = torch.tensor(obs)[:, None, :, :]
    print(obs.shape)
    new = torch.cat([obs, obs, obs, obs], dim=1)
    print(new.shape)

    agent = vPPO(7)
    out = agent.select_action(new)
    print(out.shape)

    act = [0, 0, 0, 0, 0, 0, 0, 0]
    act2 = torch.zeros(8)
    act2 = act2.int().tolist()
    _, _, _, _, info = envs.step(act2)
    _, _, _, _, info = envs.step(act2)
    print(info["episode_frame_number"])
    for item in info:
        print(item)
