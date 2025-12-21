import os
from src.maps import get_map
from src.experiments import run_experiment_suite
from src.plotting import plot_throughput, plot_returns, plot_policy_distance
from src.hyperparameter_study import run_sensitivity_study

# --- Configuration ---
MAP_SIZE = 6            
N_RUNS = 10             
EPISODES = 5000         

# Environment Parameters
ENV_CONFIG = {
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

# Training Parameters
TRAIN_CONFIG = {
    'episodes': EPISODES,
    'agent_params': {
        'epsilon': 0.1,
        'alpha': 0.1,
        'gamma': 1.0,
        'min_epsilon': 0.01,
        'epsilon_decay': 0.9995 
    }
}

def main():
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # --- PART 1: Main Comparison (MC vs SARSA) ---
    algorithms_to_run = ["MC", "SARSA"]
    shaping_types = ["baseline", "step_cost", "potential", "custom"]

    for algo in algorithms_to_run:
        print(f"\n{'='*40}")
        print(f"STARTING ANALYSIS FOR: {algo}")
        print(f"{'='*40}")
        
        # 1. Run Experiments
        results_df, runs_data, best_agent = run_experiment_suite(
            agent_type=algo,
            shaping_types=shaping_types,
            env_config=ENV_CONFIG,
            train_config=TRAIN_CONFIG,
            n_runs=N_RUNS
        )

        print(f"\nGenerating plots for {algo}...")
        
        # 2. Generate Plots
        plot_throughput(results_df, title_suffix=f"({algo})")
        plot_returns(results_df, title_suffix=f"({algo})")

        # Optional Plot (Commented out)
        # if best_agent is not None:
        #     plot_policy_distance(runs_data, best_agent, title_suffix=f"({algo})")
        
        print(f"Finished {algo}. Check 'results' folder.\n")

    # --- PART 2: Hyperparameter Sensitivity Study ---
    # מפעיל את הבדיקה על כל הפרמטרים ועבור שני הסוכנים
    run_sensitivity_study()

    print("\nALL DONE! Check the 'results' folder for all plots.")

if __name__ == "__main__":
    main()