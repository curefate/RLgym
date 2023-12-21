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
        state, info = env.reset()
        state = model.img2tensor(state)[None, :, :, :]
        print(state.shape)
        next_state = None
        done = False
        action = None
        total_reward = 0
        skip_count = 0
        # run once time
        while not done:
            # 前skip次，填满state
            if skip_count < args.skip:
                mid_state, _, terminated, truncated, _ = env.step(0)
                mid_state = model.img2tensor(mid_state)[None, :, :, :]
                if terminated or truncated:
                    done = True
                state = torch.cat([state, mid_state], dim=0)
            # state已满，填skip-1个next_state
            elif (skip_count + 1) % args.skip != 0:
                action = agent.select_action(state)
                mid_state, _, terminated, truncated, _ = env.step(action)
                mid_state = model.img2tensor(mid_state)[None, :, :, :]
                if terminated or truncated:
                    done = True
                if next_state is None:
                    next_state = mid_state
                else:
                    next_state = torch.cat([next_state, mid_state], dim=0)
            # 填满最后一次next_state，并记录
            else:
                action = agent.select_action(state)
                mid_state, reward, terminated, truncated, _ = env.step(action)
                mid_state = model.img2tensor(mid_state)[None, :, :, :]
                if terminated or truncated:
                    done = True
                if next_state is None:
                    next_state = mid_state
                else:
                    next_state = torch.cat([next_state, mid_state], dim=0)

                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)

                # go next
                state = next_state
                next_state = None
                # log reward
                total_reward += reward

            # 如果在skip内结束，则上一次记录的done改为true
            if done is True and skip_count + 1 % args.skip != 0:
                transition_dict['dones'][len(transition_dict['dones']) - 1] = True

            skip_count += 1

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
        "--end_iter", type=int, default=5001, help="end_iter"
    )
    parser.add_argument(
        "--start_iter", type=int, default=0, help="start_iter"
    )
    parser.add_argument(
        "--skip", type=int, default=4, help="times of frame skip"
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
        "--epochs", type=int, default=100, help="how many times of one data trained"
    )
    args = parser.parse_args()
    args.device = 'cpu'
    if torch.cuda.is_available():
        args.device = 'cuda'

    env = gym.make('ALE/Assault-v5', render_mode=None)
    model = vPPO(env.action_space.n, args.device, args.lr)
    if args.path != '':
        model.load(args.path)

    train(model, env, args)
    env.close()
