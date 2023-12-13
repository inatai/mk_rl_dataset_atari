import gymnasium as gym
import math
import time
import random
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple, deque
from itertools import count
from models.CNNDQN import DeepQNetworkCNN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


env_name = "Breakout"




options = {}
if env_name == 'Breakout':
    env = gym.make("ALE/Breakout-v5")
    options = {
        'max_episode': 1000000,
        'max_memory_size': 10000,
        'n_frame_stack': 4,
        'screen_size': 84,
        'batch': 128,
        'gamma': 0.99,
        'eps_start': 0.9, # ε-greedyに使用
        'eps_end': 0.05,
        'eps_decay': 1000,
        'tau': 0.005, # ターゲットネットワークに使用
        'lr': 1e-3,
        'done_score': None, # このスコアに到達したら終了 Noneならmax_episodeまで学習する
        # 'done_episodes': [100, 200, 300, 400, 500]
        'done_episodes': 10000


    }
env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=1, screen_size=options['screen_size'], terminal_on_life_loss=False,
                                        grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
env = gym.wrappers.FrameStack(env, options['n_frame_stack'])

save_folder = f'data/weight/{env_name}/CNNDQN'
if not os.path.exists(save_folder): os.makedirs(save_folder)
# 最後に保存されるpath
if options['done_score']:
    save_model_name = f'{env_name}-{device}-score{options["done_score"]}-CNNDQN.pth'
elif options['done_episodes']:
    save_model_name = f'{env_name}-{device}-done_maxiter-CNNDQN.pth'
else:
    save_model_name = f'{env_name}-{device}-{options["max_episode"]}epo-CNNDQN.pth'
save_model_path = f'{save_folder}/{save_model_name}'




def main():
    start_time = time.time()

    done_episode_counter = 0
    for i_episode in range(options['max_episode']):
        state, info = env.reset()

        # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        state = torch.from_numpy(np.array(state)).float().to(device).unsqueeze(0)
        reward_sum = 0
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            reward_sum += reward
            
            reward = torch.tensor([reward], device=device)

            if terminated:
                next_state = None
            else:
                next_state = torch.from_numpy(np.array(observation)).float().to(device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = torch.add(
                    torch.mul(policy_net_state_dict[key], options['tau']),
                    torch.mul(target_net_state_dict[key], 1 - options['tau'])
                )
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                if i_episode%10 == 0:
                    seconds = int(time.time() - start_time)
                    minutes, seconds = divmod(seconds, 60)
                    hours, minutes = divmod(minutes, 60)
                    print(f'epi{i_episode} : [reward:{reward_sum},  episode_len:{t+1}], elapsed time:{hours:02}:{minutes:02}:{seconds:02}')
                break
        


        # この終了のさせ方だと、スコアだけ出して学習しないまま終了している可能性がある
        if options['done_score'] and reward_sum >= options['done_score']:
            break
        if options['done_episodes'] and i_episode % 10000 == 0:
                print(f'episode:{i_episode} model saved')
                done_epi_save_name = f'{env_name}-{device}-done_epi{i_episode}-CNNDQN.pth'
                done_episodes_save_path = f'{save_folder}/{done_epi_save_name}'
                torch.save(target_net, done_episodes_save_path)
                done_episode_counter += 1
        

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()
    torch.save(target_net, save_model_path)



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

# Get number of actions from gym action space
action_dim = env.action_space.n
state_dim = env.observation_space.shape

policy_net = DeepQNetworkCNN(state_dim, action_dim).to(device)
target_net = DeepQNetworkCNN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=options['lr'], amsgrad=True)
memory = ReplayMemory(options['max_memory_size'])


steps_done = 0


#ε-greedy法に従って行動を選択(エピソードが進むごとに探索の割合を低く)
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = options['eps_end'] + (options['eps_start'] - options['eps_end']) * \
        math.exp(-1. * steps_done / options['eps_decay'])
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)



episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < options['batch']:
        return
    transitions = memory.sample(options['batch'])

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(options['batch'], device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * options['gamma']) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()




if __name__ == "__main__":
    main()