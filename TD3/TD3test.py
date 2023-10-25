from TD3 import TD3, scale_action
import torch
import gymnasium as gym

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2', render_mode="human")
    state, info = env.reset()

    device = "cuda"
    model = TD3(env.observation_space.shape[0], env.action_space.shape[0], device)
    model.load("LunarLanderContinuous-v2/checkpoint/000400.pt")
    done = False
    i = 0
    # print(env.action_space.low)
    # print(env.action_space.high)
    # print(env.action_space.sample())
    while not done:
        action = model.select_action(state, train=False)
        action = scale_action(action, low=env.action_space.low, high=env.action_space.high)
        state, reward, terminated, truncated, info = env.step(action)
        print("{}th steps, reward={}, action={}".format(i, reward, action))
        i += 1
        if terminated or truncated:
            done = True
            break