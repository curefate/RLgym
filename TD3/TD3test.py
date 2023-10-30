from TD3 import TD3, scale_action
import torch
import gymnasium as gym

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3', render_mode="human")

    device = "cuda"
    model = TD3(env.observation_space.shape[0], env.action_space.shape[0], device)
    model.load("BipedalWalker-v3/checkpoint/003000.pt")
    for iters in range(10):
        done = False
        i = 0
        state, info = env.reset()
        while not done:
            action = model.select_action(state, train=False)
            action = scale_action(action, low=env.action_space.low, high=env.action_space.high)
            state, reward, terminated, truncated, info = env.step(action)
            print("{}th steps, reward={}, action={}".format(i, reward, action))
            i = i + 1
            if terminated or truncated:
                done = True