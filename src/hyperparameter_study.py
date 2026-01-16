import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from src.experiments import train_single_run
from src.maps import get_map
from src.plotting_utils import set_style

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
    Executes a parameter sweep, generates a FACETED sensitivity plot (separate graphs per value), 
    and identifies the best value.
    """
    print(f"\n[{agent_type}] Sweeping {param_name} values: {param_values}")
    all_results = []
    final_performances = {}

    for val in param_values:
        # Create fresh configuration copies
        env_config = DEFAULT_ENV_CONFIG.copy()
        env_config['shaping_params'] = DEFAULT_ENV_CONFIG['shaping_params'].copy()
        train_config = DEFAULT_TRAIN_CONFIG.copy()
        train_config['agent_params'] = DEFAULT_TRAIN_CONFIG['agent_params'].copy()

        # Update the specific parameter being swept
        if config_type == 'agent':
            train_config['agent_params'][inner_key] = val
        elif config_type == 'shaping':
            env_config['shaping_params'][inner_key] = val

        avg_return_metric = 0
        for run_i in range(N_RUNS_PER_VAL):
            # Run training
            df, _ = train_single_run(agent_type, SHAPING_TYPE, env_config, train_config)
            
            # === SMOOTHING ===
            # Calculates the moving average to make the graph readable
            df['smoothed_return'] = df['return'].rolling(window=200, min_periods=1).mean()
            
            temp_df = df[['episode', 'smoothed_return']].copy()
            # Rename for plotting
            temp_df.rename(columns={'smoothed_return': 'return'}, inplace=True)
            
            temp_df['Value'] = str(val)
            temp_df['run'] = run_i
            all_results.append(temp_df)
            
            # Performance metric based on end of training
            avg_return_metric += df["return"].tail(200).mean()
        
        final_performances[val] = avg_return_metric / N_RUNS_PER_VAL


    # Combine all results
    full_df = pd.concat(all_results, ignore_index=True)
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", f"sensitivity_{agent_type}_{param_name}.png")
    
    # === NEW PLOTTING LOGIC: FACETED GRAPHS ===
    set_style() # Apply the project's visual style
    
    # sns.relplot allows creating subplots (columns) for each value automatically
    g = sns.relplot(
        data=full_df,
        x="episode",
        y="return",
        col="Value",      # <--- This creates separate graphs for each parameter value
        kind="line",      # Line plot
        col_wrap=3,       # Maximum 3 graphs per row
        height=4,         # Height of each small graph
        aspect=1.2,       # Width ratio
        facet_kws={'sharey': True} # Make sure all graphs share the same Y-axis scale for fair comparison
    )
    
    # Add a main title to the entire figure
    g.fig.suptitle(f"Sensitivity Analysis: {param_name} ({agent_type})", y=1.02, fontsize=16)
    g.set_axis_labels("Episode", "Avg Return (Smoothed)")
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved faceted plot to {save_path}")

    # Identify the best parameter value
    best_val = max(final_performances, key=final_performances.get)
    print(f"Winner for {param_name}: {best_val} (Avg Return Score: {final_performances[best_val]:.3f})")
    return best_val

def find_best_hyperparameters(agent_type):
    """
    Runs the full hyperparameter optimization suite.
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