import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from src.experiments import train_single_run
from src.maps import get_map

# --- Configuration for Sweep ---
MAP_SIZE = 6
BASE_ENV_CONFIG = {
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

# We will sweep ALPHA (Learning Rate) as requested in the PDF
HYPERPARAMS_TO_SWEEP = [0.05, 0.1, 0.2]
AGENT_TYPE = "SARSA" # Or "MC"
SHAPING_TYPE = "potential" # Use one shaping method for fair comparison
N_EPISODES = 5000
N_RUNS_PER_VAL = 5 # Average over 5 runs to get smooth curves

def run_sensitivity_study():
    print(f"Starting Hyperparameter Sensitivity Study for {AGENT_TYPE} varying Alpha...")
    
    all_results = []

    for alpha_val in HYPERPARAMS_TO_SWEEP:
        print(f"Testing alpha = {alpha_val}...")
        
        train_config = {
            'episodes': N_EPISODES,
            'agent_params': {
                'epsilon': 0.1,
                'alpha': alpha_val, # THIS IS THE VARIABLE
                'gamma': 1.0
            }
        }

        for run_i in range(N_RUNS_PER_VAL):
            df, _ = train_single_run(AGENT_TYPE, SHAPING_TYPE, BASE_ENV_CONFIG, train_config)
            
            # Keep only necessary columns to save memory/time
            df = df[['episode', 'return']].copy()
            df['alpha'] = alpha_val
            df['run'] = run_i
            all_results.append(df)

    # Combine all data
    full_df = pd.concat(all_results, ignore_index=True)

    # --- Plotting ---
    print("Generating Sensitivity Plot...")
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    
    # Plot line with standard deviation (ci='sd')
    sns.lineplot(
        data=full_df, 
        x="episode", 
        y="return", 
        hue="alpha", 
        palette="viridis",
        errorbar='sd'
    )
    
    plt.title(f"Hyperparameter Sensitivity: Learning Rate (Alpha) - {AGENT_TYPE}")
    plt.xlabel("Episode")
    plt.ylabel("Return per Episode")
    plt.legend(title="Alpha Value")
    
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", f"sensitivity_alpha_{AGENT_TYPE}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    run_sensitivity_study()