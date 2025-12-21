import numpy as np
import pandas as pd
import gymnasium as gym
from tqdm import tqdm
from src.mc_agent import MonteCarloAgent
from src.sarsa_agent import SarsaAgent
from src.wrappers import RewardShapingWrapper


def train_single_run(agent_type, shaping_type, env_config, train_config):
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

    if agent_type == "MC":
        agent = MonteCarloAgent(n_actions, n_states, **train_config['agent_params'])
    elif agent_type == "SARSA":
        agent = SarsaAgent(n_actions, n_states, **train_config['agent_params'])
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
                # ✔️ include terminal transition
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

    return pd.DataFrame(history), agent


def run_experiment_suite(agent_type, shaping_types, env_config, train_config, n_runs=20):
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
