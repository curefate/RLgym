from DQN import DQN_Network
import torch
import gymnasium as gym

if __name__ == '__main__':
    device = "cuda"
    model = DQN_Network(4, 2).to(device)
    ckpt = torch.load("Cartpole/checkpoint/009000.pt", map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt["model"])

    env = gym.make('CartPole-v1', render_mode="human")
    state, info = env.reset()
    for iters in range(10):
        done = False
        state, info = env.reset()
        total_reward = 0
        while not done:
            action = model(torch.tensor(state).view(-1, 4).to(device)).max(1)[1].item()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                done = True

        print("{}th test, total reward={}".format(iters, total_reward))
