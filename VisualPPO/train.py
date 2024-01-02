import argparse
import random
import time

import gymnasium as gym
import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default='', help="path of model"
    )
    parser.add_argument(
        "--gym_id", type=str, default='ALE/Assault-v5', help=""
    )
    parser.add_argument(
        "--run_name", type=str, default='default_run_name', help=""
    )
    parser.add_argument(
        "--already_trained_times", type=int, default=0, help=""
    )
    parser.add_argument(
        "--seed", type=int, default=1, help=""
    )
    # hyper parameters
    parser.add_argument(
        "--num_envs", type=int, default=8, help="environments number"
    )
    parser.add_argument(
        "--total_time_steps", type=int, default=10000000, help=""
    )
    parser.add_argument(
        "--num_rollout_step", type=int, default=128, help=""
    )
    parser.add_argument(
        "--minibatch_size", type=int, default=4, help=""
    )
    parser.add_argument(
        "--clip_coef", type=float, default=.1, help=""
    )
    parser.add_argument(
        "--lossv_coef", type=int, default=.5, help=""
    )
    parser.add_argument(
        "--losse_coef", type=int, default=.01, help=""
    )
    parser.add_argument(
        "--norm_adv", type=bool, default=True, help="if normalize advantage function"
    )
    parser.add_argument(
        "--clip_vloss", type=bool, default=True, help="if clip value loss"
    )
    parser.add_argument(
        "--anneal_lr", type=bool, default=True, help="if anneal learning rate"
    )
    parser.add_argument(
        "--end_iter", type=int, default=3001, help="end_iter"
    )
    parser.add_argument(
        "--start_iter", type=int, default=0, help="start_iter"
    )
    parser.add_argument(
        "--lr", type=float, default=2.5e-4, help="learning rate"
    )
    parser.add_argument(
        "--gamma", type=float, default=.99, help="discount constant"
    )
    parser.add_argument(
        "--lmbda", type=float, default=.95, help="scaler of GAE"
    )
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_rollout_step)
    args.minibatch_size = int(args.batch_size // args.minibatch_size)
    args.device = 'cpu'
    if torch.cuda.is_available():
        args.device = 'cuda'

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # env setup
    run_name = f"{args.gym_id}__{args.run_name}__{args.seed}__{int(time.time())}"
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )