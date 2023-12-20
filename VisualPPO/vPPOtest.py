
import torch
import gymnasium as gym

from VisualPPO.vPPO import vPPO, fetch_image

if __name__ == '__main__':
    env = gym.make('LunarLander-v2', render_mode="human")

    device = "cuda"
    model = vPPO(512, env.action_space.n, device)
    model.load("LunarLander-v2/checkpoint/005000.pt")
    for iters in range(10):
        done = False
        _, info = env.reset()
        screen = model.img2tensor(fetch_image())
        state = model.screen2state(screen)
        total_reward = 0
        while not done:
            action = model.select_action(state)
            _, reward, terminated, truncated, info = env.step(action)
            screen = model.img2tensor(fetch_image())
            state = model.screen2state(screen)
            total_reward += reward
            if terminated or truncated:
                done = True

        print("{}th test, total reward={}".format(iters, total_reward))
