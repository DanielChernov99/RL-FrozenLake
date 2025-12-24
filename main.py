import os
import copy
from src.maps import get_map
from src.experiments import run_experiment_suite
from src.plotting import plot_throughput, plot_returns
from src.hyperparameter_study import find_best_hyperparameters

# --- Configuration ---
MAP_SIZE = 6            
N_RUNS = 20             
EPISODES = 5000         

# === המתג שביקשת ===
# True = בצע סריקה (Sweep), מצא את ה-Alpha הכי טוב, והשתמש בו (עם Gamma=1.0).
# False = אל תבצע סריקה, השתמש בערכים המוגדרים למטה ב-TRAIN_CONFIG.
USE_OPTIMIZED_PARAMS = True 

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

# Base Training Parameters (Defaults)
TRAIN_CONFIG = {
    'episodes': EPISODES,
    'agent_params': {
        'epsilon': 0.1,
        'alpha': 0.1,   # ערך ברירת מחדל (אם המתג כבוי)
        'gamma': 1.0,   # חובה להיות 1.0 בניסוי הראשי
        'min_epsilon': 0.01,
        'epsilon_decay': 0.9995 
    }
}

def main():
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    algorithms_to_run = ["MC", "SARSA"]
    
    # רשימת כל סוגי העיצוב (כולל החדשים שהוספנו)
    shaping_types = ["baseline", "step_cost", "potential", "custom_safety", "custom_advanced"]

    for algo in algorithms_to_run:
        print(f"{'='*40}")
        print(f"PREPARING ALGORITHM: {algo}")
        print(f"{'='*40}")

        # יצירת עותק עמוק של הקונפיגורציה כדי לא לדרוס ריצות קודמות
        current_train_config = copy.deepcopy(TRAIN_CONFIG)

        # --- לוגיקת המתג ---
        if USE_OPTIMIZED_PARAMS:
            print(f"b[SWITCH ON] Running sensitivity study to find best Alpha for {algo}...")
            # הפונקציה הזו כבר מוגדרת להחזיר Gamma=1.0 קבוע
            best_params = find_best_hyperparameters(algo)
            
            # עדכון הפרמטרים עם מה שמצאנו
            current_train_config['agent_params'].update(best_params)
            print(f">>> Selected Optimized Parameters: {best_params}")
        else:
            print(f"[SWITCH OFF] Using DEFAULT parameters from TRAIN_CONFIG.")
            print(f">>> Current Parameters: {current_train_config['agent_params']}")
        
        # --- STEP 2: Main Comparison ---
        print(f"\nSTARTING MAIN EXPERIMENTS FOR: {algo}")
        
        results_df, runs_data, best_agent = run_experiment_suite(
            agent_type=algo,
            shaping_types=shaping_types,
            env_config=ENV_CONFIG,
            train_config=current_train_config,
            n_runs=N_RUNS
        )

        print(f"\nGenerating plots for {algo}...")
        
        plot_throughput(results_df, title_suffix=f"({algo})")
        
        # שימוש בפונקציית ה-Returns המתוקנת (עם החלקה)
        plot_returns(results_df, title_suffix=f"({algo})")

        print(f"Finished {algo}. Check 'results' folder.\n")

    print("\nALL DONE! All experiments completed.")

if __name__ == "__main__":
    main()