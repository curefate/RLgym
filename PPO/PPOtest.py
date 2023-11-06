from PPO import PPO
import torch
import gymnasium as gym

if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode="human")

    device = "cuda"
    model = PPO(env.observation_space.shape[0], 2, device)
    model.load("MountainCar-v0/checkpoint/003000.pt")
    for iters in range(10):
        done = False
        i = 0
        state, info = env.reset()
        while not done:
            action = model.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            print("{}th steps, reward={}, action={}".format(i, reward, action))
            i = i + 1
            if terminated or truncated:
                done = True