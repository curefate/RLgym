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
    for i in range(99999):
        action = model(torch.tensor(state).view(-1, 4).to(device)).max(1)[1].item()
        state, reward, terminated, truncated, info = env.step(action)
        print("{}th steps, reward={}, action={}".format(i, reward, action))
        if terminated or truncated:
            break
