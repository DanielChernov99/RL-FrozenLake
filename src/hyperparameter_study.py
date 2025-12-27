import pandas as pd
import numpy as np
import os
from src.experiments import train_single_run
from src.maps import get_map
from src.plotting_utils import plot_generic_lineplot

"""
Hyperparameter Study Module

This script is responsible for running sensitivity analysis on agent parameters (like Alpha and Gamma).
It executes sweeps over defined ranges, plots the results, and returns the optimal configuration.
"""

# --- Configuration for Sweep ---
MAP_SIZE = 6
N_EPISODES = 5000   
N_RUNS_PER_VAL = 5 
SHAPING_TYPE = "potential" 

# Values to test
SWEEP_CONFIG = {
    "alpha": [0.05, 0.1, 0.2],
    "gamma": [0.95, 0.99, 1.0],
}

DEFAULT_ENV_CONFIG = {
    'desc': get_map(MAP_SIZE),
    'is_slippery': True,
    'shaping_params': {
        'gamma': 1.0,
        'step_cost_c': -0.001,
        'potential_beta': 1.0,
        'custom_c': 0.5,
        'decay_rate': 1.0 
    }
}

DEFAULT_TRAIN_CONFIG = {
    'episodes': N_EPISODES,
    'agent_params': {
        'epsilon': 0.1,
        'alpha': 0.1,
        'gamma': 1.0,
        'min_epsilon': 0.01,
        'epsilon_decay': 0.9995
    }
}

def run_single_sweep_and_plot(agent_type, param_name, param_values, config_type, inner_key):
    """
    Executes a parameter sweep, generates a sensitivity plot, and identifies the best value.
    Args:
        agent_type (str): The agent to test ("MC" or "SARSA").
        param_name (str): Display name for the parameter (e.g., "Alpha").
        param_values (list): List of values to iterate over.
        config_type (str): Which config dict to update ("agent" or "shaping").
        inner_key (str): The specific dictionary key to update (e.g., "alpha").
    Returns:
        float/int: The parameter value that yielded the highest average throughput.
    """
    print(f"\n[{agent_type}] Sweeping {param_name} values: {param_values}")
    all_results = []
    final_performances = {}

    for val in param_values:
        env_config = DEFAULT_ENV_CONFIG.copy()
        env_config['shaping_params'] = DEFAULT_ENV_CONFIG['shaping_params'].copy()
        train_config = DEFAULT_TRAIN_CONFIG.copy()
        train_config['agent_params'] = DEFAULT_TRAIN_CONFIG['agent_params'].copy()

        if config_type == 'agent':
            train_config['agent_params'][inner_key] = val
        elif config_type == 'shaping':
            env_config['shaping_params'][inner_key] = val

        avg_throughput = 0
        for run_i in range(N_RUNS_PER_VAL):
            df, _ = train_single_run(agent_type, SHAPING_TYPE, env_config, train_config)
            
            temp_df = df[['episode', 'throughput']].copy()
            temp_df['Value'] = str(val)
            temp_df['run'] = run_i
            all_results.append(temp_df)
            
            avg_throughput += df["throughput"].iloc[-1]
        
        final_performances[val] = avg_throughput / N_RUNS_PER_VAL


    full_df = pd.concat(all_results, ignore_index=True)
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", f"sensitivity_{agent_type}_{param_name}.png")
    
    plot_generic_lineplot(
        data_df=full_df,
        x_col="episode",
        y_col="throughput",
        hue_col="Value",
        title=f"Sensitivity: {param_name} ({agent_type})",
        xlabel="Episode",
        ylabel="Throughput",
        save_path=save_path
    )

    best_val = max(final_performances, key=final_performances.get)
    print(f"Winner for {param_name}: {best_val} (Avg Throughput: {final_performances[best_val]:.3f})")
    return best_val

def find_best_hyperparameters(agent_type):
    """
    Runs the full hyperparameter optimization suite.
    It performs sequential sweeps and returns a dictionary with the optimal parameters found.
    Args:
        agent_type (str): The agent to optimize.

    Returns:
        dict: A dictionary containing the best 'alpha' and 'gamma' found.
    """
    print(f"{'='*40}")
    print(f"STARTING HYPERPARAMETER TUNING FOR {agent_type}")
    print(f"{'='*40}")

    best_params = {}

    # 1. Sweep Alpha
    best_alpha = run_single_sweep_and_plot(agent_type, "Alpha", SWEEP_CONFIG["alpha"], 'agent', 'alpha')
    best_params['alpha'] = best_alpha

    # 2. Sweep Gamma
    best_gamma = run_single_sweep_and_plot(agent_type, "Gamma", SWEEP_CONFIG["gamma"], 'agent', 'gamma')
    best_params['gamma'] = best_gamma
    
    print(f"\n>>> Best params found for {agent_type}: {best_params}")
    return best_params

if __name__ == "__main__":
    # Test run
    find_best_hyperparameters("SARSA")