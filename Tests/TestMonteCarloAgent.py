import gymnasium as gym
import sys
import os

# Add the current folder to the Python path
sys.path.append(os.getcwd())

from src.agents.monte_carlo_agent import MonteCarloAgent 
from src.maps import get_map
import src.rl_utils as utils

def run_test_for_size(map_name):
    print(f"\n{'='*40}")
    print(f" start testing for:  {map_name}")
    print(f"{'='*40}")

    # 1. Set parameters based on the map name/size
    if "2x2" in map_name:
        total_episodes = 500
        visualize_process = True
        view_interval = 100
        epsilon = 0.2
    elif "4x4" in map_name:
        total_episodes = 5000
        visualize_process = True
        view_interval = 1000
        epsilon = 0.1
    else:  # 6x6, 8x8 וכו'
        total_episodes = 20000
        visualize_process = False
        epsilon = 0.1

    # 2. Create environment and agent
    try:
        custom_map = get_map(map_name)
    except ValueError as e:
        print(f"Skip: {e}")
        return

    # Create a temporary environment to get the action space size
    temp_env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False)
    agent = MonteCarloAgent(
        action_space_n=temp_env.action_space.n, 
        gamma=0.9, 
        epsilon=epsilon, 
        alpha=0.1
    )
    temp_env.close()

    # 3. Training phase (varies based on visualize_process)
    if visualize_process:
        print("Small maps running with visualization during training")
        utils.train_with_visualization(map_name, agent, total_episodes, view_interval)
    else:
        print("Large map training without visualization")
        # יוצרים סביבה לאימון השקט
        env_train = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode=None)
        utils.train_agent(env_train, agent, episodes=total_episodes)
        env_train.close()
        print("Training Completed")

    # 4. Evaluate results
    print("evaluating agent performance over 100 episodes")
    env_eval = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False)
    success_rate, returns, lengths = utils.evaluate_agent(env_eval, agent, episodes=100)
    env_eval.close()

    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average return: {sum(returns)/len(returns):.3f}")
    print(f"Average length: {sum(lengths)/len(lengths):.1f} steps")


    # 5. Final visualization (only if successful)
    # Even for large maps, show one successful episode at the end
    if success_rate >= 90:
        print("success rate is good , running final visual episode")
        utils.run_visual_episode(custom_map, agent, sleep_time=0.4)
    else:
        print("Agent did not reach success rate threshold")
def main():
    # List of maps to test.
    # Now testing maps: 2x2_simple, 4x4_1hole, 6x6
    map_sizes_to_test = ["2x2_simple", "4x4_1hole", "6x6"] 
    
    for map_name in map_sizes_to_test:
        run_test_for_size(map_name)

if __name__ == "__main__":
    main()