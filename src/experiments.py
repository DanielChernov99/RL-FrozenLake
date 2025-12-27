import numpy as np
import pandas as pd
import gymnasium as gym
from tqdm import tqdm
from src.mc_agent import MonteCarloAgent
from src.sarsa_agent import SarsaAgent
from src.wrappers import RewardShapingWrapper


def train_single_run(agent_type, shaping_type, env_config, train_config):
    """
    Executes a single training session for a specific agent and reward shaping strategy.

    Args:
        agent_type (str): "MC" or "SARSA".
        shaping_type (str): The type of reward shaping to apply.
        env_config (dict): Configuration for the environment.
        train_config (dict): Configuration for training .

    Returns:
        tuple:
            - pd.DataFrame: History of the run (episodes, returns, throughput).
            - Agent: The trained agent object.
    """
    env = gym.make(
        "FrozenLake-v1",
        desc=env_config['desc'],
        is_slippery=env_config['is_slippery']
    )
    env = RewardShapingWrapper(
        env,
        shaping_type=shaping_type,
        **env_config['shaping_params']
    )

    n_actions = env.action_space.n
    n_states = env.observation_space.n

    agent_params = train_config['agent_params'].copy()
    
    # "Extract" parameters that belong to the training loop logic, not the agent itself.
    # The 'pop' function returns the value and removes it from the dictionary, 
    # ensuring the agent class doesn't receive unexpected arguments.
    min_epsilon = agent_params.pop('min_epsilon', 0.01)
    epsilon_decay = agent_params.pop('epsilon_decay', 0.9995)

    if agent_type == "MC":
        agent = MonteCarloAgent(n_actions, n_states, **agent_params)
    elif agent_type == "SARSA":
        agent = SarsaAgent(n_actions, n_states, **agent_params)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    history = {
        "episode": [],
        "return": [],
        "throughput": [],
        "q_snapshots": []
    }

    cumulative_wins = 0
    SNAPSHOT_INTERVAL = 50

    for ep in range(1, train_config['episodes'] + 1):
        obs, _ = env.reset()
        done, truncated = False, False
        total_original_reward = 0.0
        episode_data_mc = []

        action = agent.get_action(obs)

        while True:
            next_state, shaped_reward, done, truncated, info = env.step(action)

            original_reward = info.get("original_reward", shaped_reward)
            total_original_reward += original_reward

            if agent_type == "SARSA":
                next_action = agent.get_action(next_state)
                agent.update(
                    obs,
                    action,
                    shaped_reward,
                    next_state,
                    next_action,
                    done
                )
                obs, action = next_state, next_action

                if done or truncated:
                    break

            elif agent_type == "MC":
                episode_data_mc.append((obs, action, shaped_reward))
                obs = next_state

                if done or truncated:
                    break
                else:
                    action = agent.get_action(obs)

        if agent_type == "MC" and not truncated:
            agent.update(episode_data_mc)

        is_success = 1 if (done and original_reward > 0) else 0
        cumulative_wins += is_success

        history["episode"].append(ep)
        history["return"].append(total_original_reward)
        history["throughput"].append(cumulative_wins / ep)

        if ep % SNAPSHOT_INTERVAL == 0 or ep == 1:
            history["q_snapshots"].append(agent.Q.copy())
        else:
            history["q_snapshots"].append(None)

        if agent_type == "MC":
            agent.decay_epsilon(epsilon_decay, min_epsilon)

    return pd.DataFrame(history), agent


def run_experiment_suite(agent_type, shaping_types, env_config, train_config, n_runs=20):
    """
    Orchestrates a full suite of experiments for a specific agent type across multiple 
    reward shaping strategies.

    Args:
        agent_type (str): The algorithm to use ("MC" or "SARSA").
        shaping_types (list): A list of shaping strategy names to test.
        env_config (dict): Environment configuration.
        train_config (dict): Training configuration.
        n_runs (int): Number of independent runs per shaping strategy to ensure statistical significance.

    Returns:
        tuple:
            - pd.DataFrame: Aggregated results from all runs (excluding Q-tables).
            - dict: Raw storage of full run data per shaping type.
            - Agent: The single best performing agent instance across all runs.
    """
    all_results = []
    best_agent_overall = None
    best_performance = -np.inf
    runs_data_storage = {}

    print(f"--- Starting Experiment Suite for {agent_type} ---")

    for shaping in shaping_types:
        runs_data_storage[shaping] = []
        print(f"Running {n_runs} runs for shaping: {shaping}...")

        for run_id in tqdm(range(n_runs)):
            df, final_agent = train_single_run(
                agent_type,
                shaping,
                env_config,
                train_config
            )

            final_throughput = df["throughput"].iloc[-1]
            if final_throughput > best_performance:
                best_performance = final_throughput
                best_agent_overall = final_agent

            df["shaping"] = shaping
            df["run_id"] = run_id
            df["algorithm"] = agent_type

            runs_data_storage[shaping].append(df)
            all_results.append(df.drop(columns=["q_snapshots"]))

    return (
        pd.concat(all_results, ignore_index=True),
        runs_data_storage,
        best_agent_overall
    )