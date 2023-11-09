from PPO import PPO
import torch
import gymnasium as gym

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode="human")

    device = "cuda"
    model = PPO(env.observation_space.shape[0], env.action_space.n, device)
    model.load("CartPole-v1/checkpoint/002000.pt")
    for iters in range(10):
        done = False
        state, info = env.reset()
        total_reward = 0
        while not done:
            action = model.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                done = True

        print("{}th test, total reward={}".format(iters, total_reward))
