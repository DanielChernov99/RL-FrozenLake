import matplotlib.pyplot as plt
import seaborn as sns
import os

def set_style():
    """מגדיר את העיצוב הכללי לכל הגרפים בפרויקט"""
    sns.set_theme(style="darkgrid")
    plt.rcParams["figure.figsize"] = (10, 6)

def plot_generic_lineplot(data_df, x_col, y_col, hue_col, title, xlabel, ylabel, save_path):
    """
    פונקציית עזר גנרית ליצירת גרף קו עם סטיית תקן.
    משמשת גם את main וגם את hyperparameter_study.
    """
    set_style()
    plt.figure()
    
    # errorbar='sd' calculates Standard Deviation automatically
    sns.lineplot(
        data=data_df, 
        x=x_col, 
        y=y_col, 
        hue=hue_col, 
        palette="viridis", 
        errorbar='sd'
    )
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # עיצוב המקרא (Legend) שיהיה קריא יותר
    if hue_col:
        plt.legend(title=hue_col.replace("_", " ").title())
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")