"""
Deep diagnostic to understand Monte Carlo learning on 4x4
"""
import gymnasium as gym
import sys
import os

sys.path.append(os.getcwd())

from src.agents.monte_carlo_agent import MonteCarloAgent 
from src.maps import get_map

def run_diagnostic_episodes(map_name, num_episodes=100):
    """
    Run a few episodes and print detailed debug info
    """
    custom_map = get_map(map_name)
    print(f"Map:\n{chr(10).join(custom_map)}\n")
    
    # Create agent
    env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False)
    agent = MonteCarloAgent(action_space_n=env.action_space.n, gamma=1.0, epsilon=0.3, alpha=0.1)
    
    successes = 0
    total_returns = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        terminated = truncated = False
        episode_return = 0
        steps_taken = 0
        
        while not (terminated or truncated):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.store_step(state, action, reward)
            episode_return += reward
            steps_taken += 1
            state = next_state
        
        # Update agent at end of episode
        agent.update()
        
        if episode_return > 0:
            successes += 1
        total_returns += episode_return
        
        # Print progress
        if (episode + 1) % 20 == 0 or episode < 5:
            num_states = sum(1 for s in agent.q_table if agent.q_table[s].max() > 0)
            avg_return = total_returns / (episode + 1)
            success_rate = (successes / (episode + 1)) * 100
            print(f"Episode {episode+1:3d}: Return={episode_return}, States_learned={num_states:2d}, "
                  f"Avg_return={avg_return:.3f}, Success_rate={success_rate:.1f}%, Steps={steps_taken}")
            
            # Show Q-values for starting state
            if agent.q_table[0].max() > 0:
                print(f"           Q-values for state 0: {agent.q_table[0]}")
    
    env.close()
    
    print(f"\n=== Final Statistics ===")
    print(f"Total successes: {successes}/{num_episodes}")
    print(f"Success rate: {(successes/num_episodes)*100:.1f}%")
    print(f"Average return: {total_returns/num_episodes:.3f}")
    print(f"Total unique states visited: {len(agent.q_table)}")
    
    # Show top states
    print(f"\nTop 5 states by max Q-value:")
    top_states = sorted(agent.q_table.items(), key=lambda x: x[1].max(), reverse=True)[:5]
    for state, q_vals in top_states:
        print(f"  State {state}: max_Q={q_vals.max():.4f}, Q={q_vals}")

if __name__ == "__main__":
    print("="*70)
    print("TEST 1: 4x4_1hole with 100 episodes")
    print("="*70)
    run_diagnostic_episodes("4x4_1hole", num_episodes=100)
    
    print("\n" + "="*70)
    print("TEST 2: 4x4_1hole with 500 episodes")
    print("="*70)
    run_diagnostic_episodes("4x4_1hole", num_episodes=500)
