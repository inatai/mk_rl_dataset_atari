from collections import deque
import math
import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.CNNDQN import DeepQNetworkCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_name = 'Breakout'
save_folder = f'data/weight/{env_name}/CNNDQN'
if not os.path.exists(save_folder): os.makedirs(save_folder)


class ReplayMemory:
    def __init__(self, capacity):
        self.maxlen = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        idxs = np.random.choice(np.arange(len(self.memory)), size=batch_size)
        return [self.memory[idx] for idx in idxs]

    def __len__(self):
        return len(self.memory)


# Q-learning agent:
# initialize Q: S x A -> R
# for each step of the episode:
#   choose eps-greedy action from Q
class DQNAgent:
    def __init__(self, num_actions, replay_mem_size):
        self.dqn = DeepQNetworkCNN(in_channels=4, conv1_hidden_channels=16, conv2_hidden_channels=32,
                                   fc_hidden_units=256, num_outputs=num_actions).to(device)

        self.replay_memory = ReplayMemory(replay_mem_size)

    def apply_net(self, input):
        return self.dqn(input)

    def take_greedy_action(self, obs):
        scores = self.dqn(obs)
        act = torch.argmax(scores).item()
        return act

    def add_transition(self, transition):
        self.replay_memory.push(transition)



def train_breakout():
    RAND_SEED = None

    if RAND_SEED != None:
        random.seed(RAND_SEED)
        torch.manual_seed(RAND_SEED)
        np.random.seed(RAND_SEED)

    if env_name == 'Breakout':
        env = gym.make("ALE/Breakout-v5")
    env.action_space.seed(RAND_SEED)

    # from DQN papers
    minibatch_size = 32

    max_num_steps = 1000000
    replay_mem_size = 20000
    gamma = 0.99

    # epsilon schedule: linear annealing
    epsilon_end = 1.0
    epsilon_start = 0.01
    # this is 1 million (ish? frames =? steps) in the DQN paper
    eps_anneal_length = round(0.25 * max_num_steps)

    # eps_start = 0.7
    # eps_end = 0.2
    # eps_end_step = 9
    # t = 0: 0.7
    # t = 4: 0.45
    # t = 9: 0.2

    # 0.7 - (0.5/10) * (t + 1)
    # = 0.1 * (7 - (t + 1)/2)
    #
    # eps_start + (eps_end - eps_start) * (t + 1) / (eps_end_step + 1)

    # preprocess. these are all just the defaults
    # also add the customary stack of 4 frames

    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=1, screen_size=84, terminal_on_life_loss=False,
                                          grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)

    obs, info = env.reset(seed=RAND_SEED)
    print(info)

    dqn_agent = DQNAgent(num_actions=env.action_space.n, replay_mem_size=replay_mem_size)

    optimizer = optim.Adam(dqn_agent.dqn.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    total_rewards = 0.

    for t in range(max_num_steps):
        # epsilon =  max(eps_anneal_end,
        #                eps_anneal_start + (eps_anneal_end - eps_anneal_start) * t  / eps_anneal_length)

        epsilon = ((epsilon_end - epsilon_start)/max_num_steps) * t + epsilon_start
        if t % 10000 == 0:
            print(f"\n------ on step {t=}, {epsilon=}")
            # print("now, replay memory size = ", len(dqn_agent.replay_memory))

        obs_tensor = torch.from_numpy(np.array(obs)).float().to(device)

        # take action
        if random.random() < epsilon:
            action = random.randint(0, env.action_space.n - 1)
        else:
            action = dqn_agent.take_greedy_action(obs_tensor)
        new_obs, reward, terminated, truncated, info = env.step(action)
        new_obs_tensor = torch.from_numpy(np.array(obs)).float().to(device)

        total_rewards += reward

        is_terminal = terminated or truncated
        trans = (obs_tensor, action, reward, new_obs_tensor, is_terminal)
        dqn_agent.add_transition(trans)
        # obs is an ndarray of shape
        # print(type(obs))
        # dict
        # print(type(info))

        rm_sample = dqn_agent.replay_memory.sample(batch_size=minibatch_size)

        # each is length minibatch_size (32)
        sample_states, \
            sample_actions, \
            sample_rewards, \
            sample_next_states, \
            sample_is_terminals = map(list, zip(*rm_sample))

        batch_states = torch.stack(sample_states).to(device)
        batch_actions = torch.tensor(sample_actions, device=device)
        batch_rewards = torch.tensor(sample_rewards, device=device)
        batch_next_states = torch.stack(sample_next_states).to(device)
        batch_is_terminals = torch.tensor([1. if it == True else 0. for it in sample_is_terminals], device=device)

        one_minus_bit = 1. - batch_is_terminals
        targets = batch_rewards
        expecteds = torch.zeros(targets.shape, device=device)

        net_scores = dqn_agent.apply_net(batch_states)
        net_scores_next = dqn_agent.apply_net(batch_next_states)

        # ターゲットネットのスコア
        targets += (1. - batch_is_terminals) * gamma * torch.max(net_scores_next, 1).values

        # 訓練ネットのスコア
        batch_actions = batch_actions.unsqueeze(1)
        selected_scores = net_scores.gather(1, batch_actions)
        expecteds = selected_scores.squeeze(1)

        # 損失計算
        loss = criterion(targets, expecteds)

        optimizer.zero_grad()
        loss.backward()
        # in the paper, they make all negative rewards -1, and all positive rewards +1
        # clmp_でparam.gradの値が-1～+1に収まるようにしている
        for param in dqn_agent.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        if t % 100000 == 0 and t != 0:
                save_model_name = f'{env_name}-{device}-{t}epo-CNNDQN.pth'
                save_model_path = f'{save_folder}/{save_model_name}'
                torch.save(dqn_agent.dqn, save_model_path)


        if is_terminal:
            # print(f"Resetting because {'terminated' if terminated else 'truncated'}!")
            obs, info = env.reset()

    save_model_name = f'{env_name}-{device}-{max_num_steps}epo-CNNDQN.pth'
    save_model_path = f'{save_folder}/{save_model_name}'
    torch.save(dqn_agent.dqn, save_model_path)

    env.close()

if __name__ == '__main__':
    print("hello, breakout")
    train_breakout()