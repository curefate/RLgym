import argparse
import gymnasium as gym
import torch
from tqdm import tqdm
import os.path
from torch.utils.tensorboard import SummaryWriter
from VisualPPO.vPPO import vPPO, fetch_image


def train(agent: vPPO, env, args):
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
        _, info = env.reset()
        state = model.img2tensor(fetch_image())
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            _, reward, terminated, truncated, info = env.step(action)
            next_state = model.img2tensor(fetch_image())
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
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            agent.save(args.save_path + f"/{str(idx).zfill(6)}.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default='', help="path of model"
    )
    parser.add_argument(
        "--save_path", type=str, default='LunarLander-v2/checkpoint', help="path to save"
    )
    parser.add_argument(
        "--logs_path", type=str, default='LunarLander-v2/logs', help="path to logs"
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

    env = gym.make('LunarLander-v2', render_mode="human")
    model = vPPO(512, env.action_space.n, args.device, args.lr)
    if args.path != '':
        model.load(args.path)

    train(model, env, args)
    env.close()