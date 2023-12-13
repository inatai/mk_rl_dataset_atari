import gymnasium as gym

import torch
import random
import cv2
from models.DDQN import Dueling_Network
from models.CNNDQN import DeepQNetworkCNN

import numpy as np

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    env_name = 'Breakout'
    save_path = 'Breakout-cuda-900000epo-CNNDQN.pth'
    test_save_model(env_name, save_path)



def test_save_model(env_name, save_path):
    if env_name == "Breakout":
        env = gym.make('ALE/Breakout-v5', render_mode='human')
    
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=1, screen_size=84, terminal_on_life_loss=False,
                                          grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)

    model = torch.load(f'data/weight/{env_name}/CNNDQN/{save_path}')
    model.eval()

    state, info = env.reset()
    old_life = info['lives'] + 1

    done = False
    rewards = 0
    while not done:
        # 環境の描画
        env.render()
        
        if old_life > info['lives']:
            old_life = info['lives']
            action = 1
        else:
            state_tensor = torch.from_numpy(np.array(state)).float().to(device)
            action = model(state_tensor).argmax().item()

        state, reward, terminated, truncated, info  = env.step(action)
        done = terminated or truncated

        rewards += reward
        print(rewards)

    env.close()

    print(rewards)





if __name__ == "__main__":
    main()