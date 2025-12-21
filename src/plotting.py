import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from src.plotting_utils import set_style, plot_generic_lineplot

def plot_throughput(data_df, title_suffix=""):
    """מייצר את גרף ה-Throughput לניסוי הראשי"""
    save_path = os.path.join("results", f"throughput_{title_suffix.replace(' ', '_')}.png")
    
    plot_generic_lineplot(
        data_df=data_df,
        x_col="episode",
        y_col="throughput",
        hue_col="shaping",
        title=f"Throughput over Episodes {title_suffix}",
        xlabel="Episode",
        ylabel="Throughput (Cumulative Success Rate)",
        save_path=save_path
    )

def plot_returns(data_df, title_suffix=""):
    """
    מייצר את גרף 4-הריבועים (FacetGrid).
    הוא נשאר כאן כי הוא ייחודי לניסוי הראשי ולא משתמש בגרף הגנרי.
    """
    set_style()
    
    g = sns.FacetGrid(data_df, col="shaping", col_wrap=2, height=4, aspect=1.5, sharey=True)
    g.map_dataframe(sns.lineplot, x="episode", y="return", errorbar='sd')
    
    g.fig.suptitle(f"Per-Episode Returns - Separated {title_suffix}", y=1.02, fontsize=16)
    g.set_axis_labels("Episode", "Return (Success Rate)")
    g.set_titles(col_template="{col_name}")
    
    # קווי עזר (1.0 ו-0.0)
    for ax in g.axes.flat:
        ax.axhline(1.0, ls='--', c='green', alpha=0.3)
        ax.axhline(0.0, ls='--', c='red', alpha=0.3)

    save_path = os.path.join("results", f"returns_separated_{title_suffix.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved separated returns plot to {save_path}")

# --- פונקציות למרחק מדיניות (אופציונלי) ---

def compute_tv_distance(policy_q, optimal_q):
    if policy_q is None: return np.nan
    best_actions_current = np.argmax(policy_q, axis=1)
    best_actions_optimal = np.argmax(optimal_q, axis=1)
    diffs = np.sum(best_actions_current != best_actions_optimal)
    return diffs / policy_q.shape[0]

def plot_policy_distance(runs_data_storage, best_agent, title_suffix=""):
    print("Calculating Policy Distances...")
    optimal_q = best_agent.Q
    dist_data = []
    
    for shaping, runs in runs_data_storage.items():
        for run_df in runs:
            snapshots = run_df[run_df["q_snapshots"].notnull()]
            for _, row in snapshots.iterrows():
                dist = compute_tv_distance(row["q_snapshots"], optimal_q)
                dist_data.append({"episode": row["episode"], "distance": dist, "shaping": shaping})
    
    df_dist = pd.DataFrame(dist_data)
    save_path = os.path.join("results", f"policy_distance_{title_suffix.replace(' ', '_')}.png")
    
    # שימוש בפונקציית העזר הגנרית
    plot_generic_lineplot(
        data_df=df_dist,
        x_col="episode",
        y_col="distance",
        hue_col="shaping",
        title=f"Policy Distribution Distance {title_suffix}",
        xlabel="Episode",
        ylabel="Distance to Best Policy (Lower is Better)",
        save_path=save_path
    )