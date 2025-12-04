import gymnasium as gym
import time
import sys
import os
from src.maps import get_map

def train_agent(env, agent, episodes=1000):
    """
    Continuous and fast training without graphics.
    Suitable for all agent types (Monte Carlo, SARSA, Q-Learning).
    """
    for i in range(episodes):
        state, _ = env.reset()
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Generic check: Does the agent need to store steps (Monte Carlo)?
            if hasattr(agent, 'store_step'):
                agent.store_step(state, action, reward)
            # Or is it an agent that learns at every step (SARSA/Q-Learning)?
            elif hasattr(agent, 'learn'):
                 agent.learn(state, action, reward, next_state, terminated)

            state = next_state
            
        # Update at the end of the episode (for Monte Carlo)
        if hasattr(agent, 'update'):
            agent.update()
        
        # Print progress every N episodes
        if (i + 1) % 1000 == 0:
            # Quick evaluation
            q_values_learned = sum(1 for state in agent.q_table if agent.q_table[state].max() > 0)
            print(f"  Episode {i+1}: {q_values_learned} states with non-zero Q-values")

def run_visual_episode(map_desc, agent, sleep_time=0.3):
    """
    Runs a single episode with graphics (render_mode='human')
    so we can visually see what the agent is doing.
    """
    # Create a temporary environment just for visualization
    env_viz = gym.make("FrozenLake-v1", desc=map_desc, is_slippery=False, render_mode="human")
    state, _ = env_viz.reset()
    terminated = truncated = False
    
    # Save original epsilon and set it to 0
    # to see the "pure knowledge" of the agent without random noise
    original_epsilon = agent.epsilon
    agent.epsilon = 0 
    
    print("\n--- Starting Visual Demo ---")
    steps = 0
    episode_return = 0
    while not (terminated or truncated):
        action = agent.get_action(state)
        q_values = agent.q_table[state]
        print(f"Step {steps}: State={state}, Best Q-value={q_values.max():.4f}, Action={action}, Q-values={q_values}")
        state, reward, terminated, truncated, _ = env_viz.step(action)
        episode_return += reward
        time.sleep(sleep_time) # Small delay to make it visible
        steps += 1
        
    env_viz.close()
    agent.epsilon = original_epsilon # Restore original epsilon
    print(f"Demo finished after {steps} steps. Total return: {episode_return}")


def train_with_visualization(map_size, agent, total_episodes, view_interval=500):
    """
    Hybrid function: Trains fast (no graphics), and occasionally pauses to show the user what is happening.
    Very useful for debugging small maps.
    """
    custom_map = get_map(map_size)
    # Training environment (no graphics - fast)
    env_train = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode=None)
    
    # Calculate how many training blocks to run
    if view_interval > total_episodes:
        view_interval = total_episodes
        
    iterations = total_episodes // view_interval
    
    print(f"Starting training for {total_episodes} episodes.")
    print(f"Visualizing agent every {view_interval} episodes.")

    for i in range(iterations):
        # 1. Fast training
        train_agent(env_train, agent, episodes=view_interval)
        
        # 2. Visual demonstration
        print(f"\nCompleted {(i+1) * view_interval}/{total_episodes} episodes.")
        print(f"Showing current agent knowledge...")
        run_visual_episode(custom_map, agent)
        
    env_train.close()

def evaluate_agent(env, agent, episodes=100):
    """
    Statistical evaluation of success rate.
    Runs without exploration (epsilon=0).
    Returns: (success_rate, episode_returns, episode_lengths)
    """
    success_count = 0
    episode_returns = []
    episode_lengths = []
    original_epsilon = agent.epsilon
    agent.epsilon = 0 
    
    for _ in range(episodes):
        state, _ = env.reset()
        terminated = truncated = False
        episode_return = 0
        steps = 0
        while not (terminated or truncated):
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            steps += 1
            if reward == 1:
                success_count += 1
        
        episode_returns.append(episode_return)
        episode_lengths.append(steps)
                
    agent.epsilon = original_epsilon
    return (success_count / episodes) * 100, episode_returns, episode_lengths