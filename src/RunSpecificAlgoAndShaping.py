import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import time
import gymnasium as gym
import numpy as np

from src.mc_agent import MonteCarloAgent
from src.sarsa_agent import SarsaAgent
from src.wrappers import RewardShapingWrapper
from src.maps import get_map
from src.experiments import modify_env_success_rate

# ==========================================
#        CONFIGURATION SECTION
# ==========================================
# 1. Choose Algorithm
ALGO_TYPE = "SARSA"   # Options: "MC", "SARSA"

# 2. Choose Shaping Strategy
SHAPING_TYPE = "potential" 
# Options: "baseline", "step_cost", "potential", "custom_safety", "custom_advanced"

# 3. Training Settings
TOTAL_EPISODES = 5000
RENDER_INTERVAL = 1000  # Show visualization every X episodes
MAP_SIZE = 6           # 6 or 8

# 4. Environment Physics
IS_SLIPPERY = True
SUCCESS_RATE = 0.7     # The fix we implemented

# 5. Agent Hyperparameters (You can tweak these)
AGENT_PARAMS = {
    'epsilon': 0.1,
    'alpha': 0.1,
    'gamma': 1.0,
    'min_epsilon': 0.01,
    'epsilon_decay': 0.9995
}

# 6. Shaping Parameters
SHAPING_PARAMS = {
    'gamma': 1.0,
    'step_cost_c': -0.001,
    'potential_beta': 1.0,
    'custom_c': 0.5,
    'decay_rate': 1.0
}
# ==========================================

def run_visualization_episode(agent, env, episode_num):
    """
    Runs a single episode in 'human' render mode to visualize the agent's behavior.
    This episode does NOT update the Q-table (it's a demonstration).
    """
    print(f"\n--- Visualizing Episode {episode_num} ---")
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    
    while not (done or truncated):
        # Select action (we use the current policy)
        # Optional: You can force epsilon=0 here if you want to see only 'Greedy' behavior
        # But keeping it normal shows you exactly how the agent behaves during training.
        action = agent.get_action(obs)
        
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        # Small delay to make it viewable (Gym usually handles this, but this ensures it)
        time.sleep(0.05) 

    status = "SUCCESS" if total_reward > 0 and done else "FAILED"
    print(f"Visualization Result: {status} (Steps: {steps})\n")


def main():
    print(f"Starting Specific Run: {ALGO_TYPE} with {SHAPING_TYPE}")
    
    # --- 1. Setup Training Environment (Fast, No Render) ---
    train_env = gym.make(
        "FrozenLake-v1", 
        desc=get_map(MAP_SIZE), 
        is_slippery=IS_SLIPPERY
    )
    if IS_SLIPPERY:
        train_env = modify_env_success_rate(train_env, success_rate=SUCCESS_RATE)
    
    train_env = RewardShapingWrapper(train_env, shaping_type=SHAPING_TYPE, **SHAPING_PARAMS)

    # --- 2. Setup Visualization Environment (Slow, Human Render) ---
    # We will assume the map is static so we can recreate it for the viz env
    viz_env = gym.make(
        "FrozenLake-v1", 
        desc=get_map(MAP_SIZE), 
        is_slippery=IS_SLIPPERY,
        render_mode="human"
    )
    # We apply the same physics modification to the visual env
    if IS_SLIPPERY:
        viz_env = modify_env_success_rate(viz_env, success_rate=SUCCESS_RATE)
    
    # Note: We don't necessarily need the RewardShapingWrapper for visualization 
    # unless we want to see shaped rewards printed, but the movement is what matters.
    # We'll wrap it just for consistency.
    viz_env = RewardShapingWrapper(viz_env, shaping_type=SHAPING_TYPE, **SHAPING_PARAMS)

    # --- 3. Initialize Agent ---
    n_actions = train_env.action_space.n
    n_states = train_env.observation_space.n
    
    # Extract decay params for manual handling
    min_epsilon = AGENT_PARAMS.pop('min_epsilon', 0.01)
    epsilon_decay = AGENT_PARAMS.pop('epsilon_decay', 0.9995)

    if ALGO_TYPE == "MC":
        agent = MonteCarloAgent(n_actions, n_states, **AGENT_PARAMS)
    elif ALGO_TYPE == "SARSA":
        agent = SarsaAgent(n_actions, n_states, **AGENT_PARAMS)
    
    # --- 4. Main Training Loop ---
    for ep in range(1, TOTAL_EPISODES + 1):
        
        # A. Check if we need to visualize THIS episode
        if ep % RENDER_INTERVAL == 0:
            run_visualization_episode(agent, viz_env, ep)
        
        # B. Standard Training (Fast)
        obs, _ = train_env.reset()
        done, truncated = False, False
        episode_data_mc = []
        
        action = agent.get_action(obs)
        
        while True:
            next_state, shaped_reward, done, truncated, _ = train_env.step(action)
            
            if ALGO_TYPE == "SARSA":
                next_action = agent.get_action(next_state)
                agent.update(obs, action, shaped_reward, next_state, next_action, done)
                obs, action = next_state, next_action
                if done or truncated:
                    break
            
            elif ALGO_TYPE == "MC":
                episode_data_mc.append((obs, action, shaped_reward))
                obs = next_state
                if done or truncated:
                    break
                else:
                    action = agent.get_action(obs)
        
        # MC Update at end of episode
        if ALGO_TYPE == "MC" and not truncated:
            agent.update(episode_data_mc)
            
        # Decay Epsilon
        if ALGO_TYPE == "MC": # MC usually decays per episode
            agent.decay_epsilon(epsilon_decay, min_epsilon)
        elif ALGO_TYPE == "SARSA": # SARSA logic in your code decayed inside? 
            # In your sarsa_agent.py you have a decay method, let's use it per episode
            agent.decay_epsilon(epsilon_decay, min_epsilon)

    print("Training Completed.")
    train_env.close()
    viz_env.close()

if __name__ == "__main__":
    main()