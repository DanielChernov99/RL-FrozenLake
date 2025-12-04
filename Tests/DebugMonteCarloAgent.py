"""
Debug script for testing Monte Carlo Agent on simple maps.
This script verifies that your Monte Carlo implementation works correctly
before moving to more complex maps with holes.
"""
import gymnasium as gym
import sys
import os

sys.path.append(os.getcwd())

from src.agents.monte_carlo_agent import MonteCarloAgent 
from src.maps import get_map
import src.rl_utils as utils

def run_debug_test(map_name, total_episodes, epsilon, alpha, gamma=1.0):
    """
    Run a single test with detailed debug output.
    """
    print(f"\n{'='*60}")
    print(f" Testing: {map_name}")
    print(f"{'='*60}")
    print(f"Episodes: {total_episodes}, Epsilon: {epsilon}, Alpha: {alpha}, Gamma: {gamma}")
    
    # Create environment
    try:
        custom_map = get_map(map_name)
    except ValueError as e:
        print(f"ERROR: {e}")
        return
    
    print(f"Map:\n{chr(10).join(custom_map)}\n")
    
    # Create agent
    temp_env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False)
    agent = MonteCarloAgent(
        action_space_n=temp_env.action_space.n,
        gamma=gamma,
        epsilon=epsilon,
        alpha=alpha
    )
    temp_env.close()
    
    # Train
    env_train = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode=None)
    utils.train_agent(env_train, agent, episodes=total_episodes)
    env_train.close()
    print(f"Training completed for {total_episodes} episodes")
    
    # Evaluate
    env_eval = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False)
    success_rate, returns, lengths = utils.evaluate_agent(env_eval, agent, episodes=100)
    env_eval.close()
    
    print(f"Success Rate (100 eval episodes): {success_rate:.2f}%")
    print(f"Average episode return: {sum(returns)/len(returns):.3f}")
    print(f"Average episode length: {sum(lengths)/len(lengths):.1f} steps")
    
    # Show Q-table
    print(f"\nQ-Table (non-zero values):")
    for state in sorted(agent.q_table.keys()):
        q_values = agent.q_table[state]
        if q_values.max() > 0:
            print(f"  State {state}: {q_values}")
    
    # Show a visual episode if success rate is good
    if success_rate >= 50:
        print(f"\nShowing visual episode...")
        utils.run_visual_episode(custom_map, agent, sleep_time=0.3)
    else:
        print(f"Success rate too low for visual demo")

def main():
    print("\n" + "="*60)
    print(" MONTE CARLO AGENT - DEBUG TESTS")
    print("="*60)
    
    # Test 1: Simplest case - 2x2 with no holes
    print("\n\n[TEST 1] 2x2 Grid with NO holes (baseline)")
    run_debug_test("2x2_simple", total_episodes=500, epsilon=0.1, alpha=0.1)
    
    # Test 2: 3x3 with no holes
    print("\n\n[TEST 2] 3x3 Grid with NO holes")
    run_debug_test("3x3_simple", total_episodes=1000, epsilon=0.1, alpha=0.1)
    
    # Test 3: 3x3 with 1 hole (easier than before)
    print("\n\n[TEST 3] 3x3 Grid WITH 1 hole")
    run_debug_test("3x3_1hole", total_episodes=3000, epsilon=0.15, alpha=0.1)
    
    # Test 4: 4x4 with 1 hole (even easier)
    print("\n\n[TEST 4] 4x4 Grid WITH 1 hole")
    run_debug_test("4x4_1hole", total_episodes=5000, epsilon=0.15, alpha=0.1)

if __name__ == "__main__":
    main()
