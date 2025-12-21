import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def set_style():
    sns.set_theme(style="darkgrid")
    plt.rcParams["figure.figsize"] = (10, 6)

def plot_throughput(data_df, title_suffix=""):
    set_style()
    plt.figure()
    # OPTIMIZATION: errorbar='sd' makes plotting instant (calculates Standard Deviation)
    # instead of the slow default Bootstrapping.
    sns.lineplot(data=data_df, x="episode", y="throughput", hue="shaping", errorbar='sd')
    plt.title(f"Throughput over Episodes {title_suffix}")
    plt.xlabel("Episode")
    plt.ylabel("Throughput (Cumulative Success Rate)")
    plt.legend(title="Shaping Strategy")
    save_path = os.path.join("results", f"throughput_{title_suffix.replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()

def plot_returns(data_df, title_suffix=""):
    set_style()
    
    # Use FacetGrid to create 4 separate plots (one for each shaping strategy)
    # col_wrap=2: Arrange plots in 2 columns (resulting in a 2x2 grid)
    # sharey=True: All plots share the same Y-axis scale (0 to 1) for easy comparison
    g = sns.FacetGrid(data_df, col="shaping", col_wrap=2, height=4, aspect=1.5, sharey=True)
    
    # Draw the line plot with standard deviation (sd) in each subplot
    # errorbar='sd' calculates the standard deviation instead of the slow bootstrapping
    g.map_dataframe(sns.lineplot, x="episode", y="return", errorbar='sd')
    
    # Titles and styling
    g.fig.suptitle(f"Per-Episode Returns - Separated {title_suffix}", y=1.02, fontsize=16)
    g.set_axis_labels("Episode", "Return (Success Rate)")
    g.set_titles(col_template="{col_name}") # Set title for each subplot based on shaping type
    
    # Add dashed reference lines at 1.0 (success) and 0.0 (failure) for better readability
    for ax in g.axes.flat:
        ax.axhline(1.0, ls='--', c='green', alpha=0.3)
        ax.axhline(0.0, ls='--', c='red', alpha=0.3)

    # Save the figure
    save_path = os.path.join("results", f"returns_separated_{title_suffix.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved separated returns plot to {save_path}")

# --- Policy Distance Functions ---

def compute_tv_distance(policy_q, optimal_q):
    """Calculates Total Variation distance between two Q-tables."""
    if policy_q is None: return np.nan
    
    # Greedy policy derivation
    best_actions_current = np.argmax(policy_q, axis=1)
    best_actions_optimal = np.argmax(optimal_q, axis=1)
    
    # Count differences
    diffs = np.sum(best_actions_current != best_actions_optimal)
    n_states = policy_q.shape[0]
    
    return diffs / n_states

def plot_policy_distance(runs_data_storage, best_agent, title_suffix=""):
    print("Calculating Policy Distances...")
    optimal_q = best_agent.Q
    
    dist_data = []
    
    for shaping, runs in runs_data_storage.items():
        for run_df in runs:
            # Only process rows with snapshots
            snapshots = run_df[run_df["q_snapshots"].notnull()]
            
            for _, row in snapshots.iterrows():
                dist = compute_tv_distance(row["q_snapshots"], optimal_q)
                
                dist_data.append({
                    "episode": row["episode"],
                    "distance": dist,
                    "shaping": shaping
                })
    
    df_dist = pd.DataFrame(dist_data)
    
    set_style()
    plt.figure()
    # Distance plot usually has fewer points, but good to optimize too
    sns.lineplot(data=df_dist, x="episode", y="distance", hue="shaping", errorbar='sd')
    plt.title(f"Policy Distribution Distance {title_suffix}")
    plt.xlabel("Episode")
    plt.ylabel("Distance to Best Policy (Lower is Better)")
    plt.legend(title="Shaping Strategy")
    
    save_path = os.path.join("results", f"policy_distance_{title_suffix.replace(' ', '_')}.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()