import matplotlib.pyplot as plt
import seaborn as sns
import os

def set_style():
    """
    Sets the global style and configuration for all plots in the project.
    """
    sns.set_theme(style="darkgrid")
    plt.rcParams["figure.figsize"] = (10, 6)

def plot_generic_lineplot(data_df, x_col, y_col, hue_col, title, xlabel, ylabel, save_path):
    """
    A generic helper function to generate line plots with Confidence Intervals (CI).
    """
    set_style()
    plt.figure()
    
    # Using 95% Confidence Interval.
    # n_boot=20 dramatically speeds up runtime compared to the default (1000)
    # while still providing a reasonable estimation of the error.
    sns.lineplot(
        data=data_df, 
        x=x_col, 
        y=y_col, 
        hue=hue_col, 
        palette="tab10", 
        errorbar=('ci', 95),
        n_boot=20  
    )
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if hue_col:
        plt.legend(title=hue_col.replace("_", " ").title())
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")