# RLgym
Some implements of reinforcement leraning include DQN, TD3. PPO and Visual PPO.
***
## Environments
The enviroments of reinforcement learning is implemented by [gymnasuim](https://gymnasium.farama.org/index.html), you can check all enviroments in their documents.
## Others
- All neural networks of each algorithms are implemented by [Pytorch](https://pytorch.org/).
- In theory, you can try any discrete environments with DQN, and any both discrete and continuous environments with TD3 and PPO, while I just written continous types. :P
- *Visual PPO* is PPO which use screenshot(image) but not normal attributes as input, it is not complete here, there is more about visual reinforcement learning in this [project](https://github.com/curefate/rl_research/tree/dev).
