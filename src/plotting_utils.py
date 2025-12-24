import matplotlib.pyplot as plt
import seaborn as sns
import os

def set_style():
    """מגדיר את העיצוב הכללי לכל הגרפים בפרויקט"""
    sns.set_theme(style="darkgrid")
    plt.rcParams["figure.figsize"] = (10, 6)

def plot_generic_lineplot(data_df, x_col, y_col, hue_col, title, xlabel, ylabel, save_path):
    """
    פונקציית עזר גנרית ליצירת גרף קו עם רווח בר סמך (CI).
    """
    set_style()
    plt.figure()
    
    # שימוש ב-Confidence Interval 95%
    # n_boot=20 מאיץ דרמטית את זמן הריצה לעומת ברירת המחדל (1000)
    sns.lineplot(
        data=data_df, 
        x=x_col, 
        y=y_col, 
        hue=hue_col, 
        palette="viridis", 
        errorbar=('ci', 95),
        n_boot=20  
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