import os
from src.maps import get_map
from src.experiments import run_experiment_suite
from src.plotting import plot_throughput, plot_returns, plot_policy_distance
from src.hyperparameter_study import find_best_hyperparameters

# --- Configuration ---
MAP_SIZE = 6            
N_RUNS = 20             
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

# Base Training Parameters (will be updated by sweep)
TRAIN_CONFIG = {
    'episodes': EPISODES,
    'agent_params': {
        'epsilon': 0.1,
        'alpha': 0.1,   # Default placeholder
        'gamma': 1.0,   # Default placeholder
        'min_epsilon': 0.01,
        'epsilon_decay': 0.9995 
    }
}

def main():
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    algorithms_to_run = ["MC", "SARSA"]
    shaping_types = ["baseline", "step_cost", "potential", "custom"]

    for algo in algorithms_to_run:
        # --- STEP 1: Hyperparameter Sweep ---
        # מוצאים את הפרמטרים הכי טובים לפני שמתחילים את הניסוי הגדול
        print(f"\nRunning sensitivity study to find best params for {algo}...")
        best_params = find_best_hyperparameters(algo)
        
        # מעדכנים את הקונפיגורציה עם המנצחים
        current_train_config = TRAIN_CONFIG.copy()
        current_train_config['agent_params'].update(best_params)
        
        print(f"\nUsing optimized parameters for {algo}: {best_params}")
        
        # --- STEP 2: Main Comparison ---
        print(f"{'='*40}")
        print(f"STARTING MAIN ANALYSIS FOR: {algo}")
        print(f"{'='*40}")
        
        results_df, runs_data, best_agent = run_experiment_suite(
            agent_type=algo,
            shaping_types=shaping_types,
            env_config=ENV_CONFIG,
            train_config=current_train_config,
            n_runs=N_RUNS
        )

        print(f"\nGenerating plots for {algo}...")
        
        plot_throughput(results_df, title_suffix=f"({algo})")
        
        # כאן הגרף יצא חלק ויפה בזכות התיקון ב-plotting.py
        plot_returns(results_df, title_suffix=f"({algo})")

        # Optional: Uncomment if you want policy distance
        # if best_agent is not None:
        #     plot_policy_distance(runs_data, best_agent, title_suffix=f"({algo})")
        
        print(f"Finished {algo}. Check 'results' folder.\n")

    print("\nALL DONE! All experiments completed.")

if __name__ == "__main__":
    main()