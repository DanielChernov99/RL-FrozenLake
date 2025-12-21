import pandas as pd
import os
from src.experiments import train_single_run
from src.maps import get_map
from src.plotting_utils import plot_generic_lineplot

# --- Configuration ---
MAP_SIZE = 6
N_EPISODES = 5000
N_RUNS_PER_VAL = 5
# בחרנו בשיטה זו כדי לבדוק את הרגישות לפרמטרים
SHAPING_TYPE = "potential" 

# --- בחירת הסוכן לניתוח (לפי ה-PDF נדרש רק אחד) ---
# אנו נתמקד ב-SARSA כפי שביקשת
AGENT_TO_STUDY = "SARSA"

# --- הערכים לבדיקה ---
SWEEP_CONFIG = {
    # בדיקת קצב למידה (Alpha) - משפיע מאוד על היציבות
    "alpha": [0.05, 0.1, 0.2],
    
    # בדיקת גורם הדעיכה (Gamma) - משפיע על ראייה לטווח רחוק
    "gamma": [0.95, 0.99, 1.0],
}

# הגדרות בסיס (קבועות)
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

def run_single_sweep(param_name, param_values, config_type, inner_key):
    """מריץ בדיקה לפרמטר אחד עבור הסוכן הנבחר"""
    print(f"\n[{AGENT_TO_STUDY}] Sweeping {param_name} values: {param_values}")
    all_results = []

    for val in param_values:
        # שכפול קונפיגורציות
        env_config = DEFAULT_ENV_CONFIG.copy()
        env_config['shaping_params'] = DEFAULT_ENV_CONFIG['shaping_params'].copy()
        train_config = DEFAULT_TRAIN_CONFIG.copy()
        train_config['agent_params'] = DEFAULT_TRAIN_CONFIG['agent_params'].copy()

        # עדכון הפרמטר הנבדק
        if config_type == 'agent':
            train_config['agent_params'][inner_key] = val
        elif config_type == 'shaping':
            env_config['shaping_params'][inner_key] = val

        # הרצת הניסויים
        for run_i in range(N_RUNS_PER_VAL):
            df, _ = train_single_run(AGENT_TO_STUDY, SHAPING_TYPE, env_config, train_config)
            
            temp_df = df[['episode', 'throughput']].copy()
            temp_df['Value'] = str(val)
            temp_df['run'] = run_i
            all_results.append(temp_df)

    # איחוד ויצירת גרף
    full_df = pd.concat(all_results, ignore_index=True)
    os.makedirs("results", exist_ok=True)
    
    save_path = os.path.join("results", f"sensitivity_{AGENT_TO_STUDY}_{param_name}.png")
    
    plot_generic_lineplot(
        data_df=full_df,
        x_col="episode",
        y_col="throughput",
        hue_col="Value",
        title=f"Sensitivity: {param_name} ({AGENT_TO_STUDY})",
        xlabel="Episode",
        ylabel="Throughput",
        save_path=save_path
    )

def run_sensitivity_study():
    print(f"{'='*40}")
    print(f"STARTING HYPERPARAMETER SENSITIVITY STUDY FOR {AGENT_TO_STUDY}")
    print(f"{'='*40}")

    # 1. Sweep Alpha (Learning Rate)
    run_single_sweep("Alpha", SWEEP_CONFIG["alpha"], 'agent', 'alpha')

    # 2. Sweep Gamma (Discount Factor)
    run_single_sweep("Gamma", SWEEP_CONFIG["gamma"], 'agent', 'gamma')
    
    print(f"\nSensitivity study finished for {AGENT_TO_STUDY}. Check 'results' folder.")

if __name__ == "__main__":
    run_sensitivity_study()