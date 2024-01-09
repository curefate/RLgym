
import torch
import gymnasium as gym

from VisualPPO.vPPO import vPPO, make_env

if __name__ == '__main__':
    env = gym.make('ALE/Galaxian-v5', render_mode="human")
    env = gym.wrappers.GrayScaleObservation(env)

    device = "cuda"
    agent = vPPO(env.action_space.n).to(device)
    agent.load("ckpt0/Galaxian-v5__default_exp_name__1__1704695912_step3686400.pt")
    for iters in range(10):
        done = False
        state, _ = env.reset()
        state = torch.tensor(state)[None, None, :, :].to(device)
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action.item())
            state = torch.tensor(state)[None, None, :, :].to(device)
            if reward != 0.:
                print(reward)
            total_reward += reward
            if terminated or truncated:
                done = True

        print("{}th test, total reward={}".format(iters, total_reward))
