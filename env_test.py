import gymnasium as gym
import time
import keyboard

env_name = "Breakout"

if env_name == "Breakout":
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    key_list = [None, "space", "right", "left"]


def main():

    _, _ = env.reset()

    done = False
    rewards = 0

    while not done:
        # 環境の描画
        env.render()


        action = N_get_action(key_list)
        _ , reward, terminated, truncated, _  = env.step(action)

        done = terminated or truncated

        rewards += reward
        
        print(reward)
    print(rewards)
    env.close()




def N_get_action(key_list):
    time.sleep(0.02)
    for i, key in enumerate(key_list):
        if key == None:
            action = i
            continue
        elif keyboard.is_pressed(key):
            action = i
            break
    return action



if __name__ == "__main__":
    main()