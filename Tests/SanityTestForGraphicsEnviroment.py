import gymnasium as gym

def random_episode():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
    obs, info = env.reset()

    terminated = truncated = False
    total_reward = 0

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print("Episode finished. Total reward =", total_reward)
    env.close()

if __name__ == "__main__":
    random_episode()
