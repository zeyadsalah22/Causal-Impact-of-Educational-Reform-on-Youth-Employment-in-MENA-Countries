"""
Visualization utilities for the MENA causal analysis.
Produces publication-quality plots for trends, parallel trends,
event studies, propensity score distributions, and balance checks.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set default style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'warning': '#D4AF37',
    'neutral': '#6C757D',
    'treated': '#E63946',
    'control': '#457B9D'
}


def plot_trends(df: pd.DataFrame,
               time_col: str,
               variables: List[str],
               group_col: Optional[str] = None,
               title: Optional[str] = None,
               figsize: Tuple[int, int] = (12, 6),
               save_path: Optional[str] = None):
    """
    Plot time trends for variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Panel dataframe
    time_col : str
        Time variable (usually 'year')
    variables : List[str]
        Variables to plot
    group_col : str, optional
        Grouping variable (e.g., 'country_code', 'treated')
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    """
    n_vars = len(variables)
    fig, axes = plt.subplots(1, n_vars, figsize=figsize)
    
    if n_vars == 1:
        axes = [axes]
    
    for idx, var in enumerate(variables):
        if group_col:
            for group in df[group_col].unique():
                group_data = df[df[group_col] == group]
                avg = group_data.groupby(time_col)[var].mean()
                axes[idx].plot(avg.index, avg.values, label=f'{group_col}={group}', linewidth=2)
            axes[idx].legend()
        else:
            avg = df.groupby(time_col)[var].mean()
            axes[idx].plot(avg.index, avg.values, linewidth=2.5, color=COLORS['primary'])
        
        axes[idx].set_xlabel(time_col.capitalize())
        axes[idx].set_ylabel(var.replace('_', ' ').title())
        axes[idx].grid(alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    plt.show()


def plot_parallel_trends(df: pd.DataFrame,
                        time_col: str,
                        outcome: str,
                        treatment_col: str,
                        treatment_time: int,
                        title: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None):
    """
    Plot parallel trends for treated and control groups.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Panel dataframe
    time_col : str
        Time variable
    outcome : str
        Outcome variable
    treatment_col : str
        Treatment indicator
    treatment_time : int
        Time of treatment
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate means by group and time
    treated_avg = df[df[treatment_col] == 1].groupby(time_col)[outcome].mean()
    control_avg = df[df[treatment_col] == 0].groupby(time_col)[outcome].mean()
    
    # Plot
    ax.plot(treated_avg.index, treated_avg.values, 
            linewidth=3, color=COLORS['treated'], label='Treated', marker='o')
    ax.plot(control_avg.index, control_avg.values, 
            linewidth=3, color=COLORS['control'], label='Control', marker='s')
    
    # Add vertical line at treatment time
    ax.axvline(x=treatment_time, color='gray', linestyle='--', 
               linewidth=2, alpha=0.7, label='Treatment Time')
    
    # Add shaded regions for pre/post
    ax.axvspan(df[time_col].min(), treatment_time, alpha=0.1, color='blue', label='Pre-Treatment')
    ax.axvspan(treatment_time, df[time_col].max(), alpha=0.1, color='red', label='Post-Treatment')
    
    ax.set_xlabel(time_col.capitalize(), fontsize=12)
    ax.set_ylabel(outcome.replace('_', ' ').title(), fontsize=12)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title('Parallel Trends Test', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    plt.show()


def plot_event_study(coefficients_df: pd.DataFrame,
                    event_time_col: str = 'Event Time',
                    coef_col: str = 'Coefficient',
                    ci_lower_col: str = 'CI_lower',
                    ci_upper_col: str = 'CI_upper',
                    title: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 6),
                    save_path: Optional[str] = None):
    """
    Plot event study coefficients with confidence intervals.
    
    Parameters:
    -----------
    coefficients_df : pd.DataFrame
        DataFrame with event time coefficients
    event_time_col : str
        Event time column name
    coef_col : str
        Coefficient column name
    ci_lower_col : str
        Lower CI column name
    ci_upper_col : str
        Upper CI column name
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    event_times = coefficients_df[event_time_col].values
    coefficients = coefficients_df[coef_col].values
    ci_lower = coefficients_df[ci_lower_col].values
    ci_upper = coefficients_df[ci_upper_col].values
    
    # Plot coefficients
    ax.plot(event_times, coefficients, 'o-', linewidth=2.5, 
            markersize=8, color=COLORS['primary'], label='Point Estimate')
    
    # Add confidence intervals
    ax.fill_between(event_times, ci_lower, ci_upper, 
                    alpha=0.2, color=COLORS['primary'], label='95% CI')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Add vertical line at treatment time (event time = 0)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, 
               alpha=0.5, label='Treatment Time')
    
    ax.set_xlabel('Event Time (Years Relative to Treatment)', fontsize=12)
    ax.set_ylabel('Treatment Effect', fontsize=12)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title('Event Study Analysis', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    plt.show()


def plot_propensity_scores(df: pd.DataFrame,
                          treatment_col: str,
                          ps_col: str = 'propensity_score',
                          title: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 6),
                          save_path: Optional[str] = None):
    """
    Plot propensity score distributions for treated and control groups.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with propensity scores
    treatment_col : str
        Treatment column name
    ps_col : str
        Propensity score column name
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    treated_ps = df[df[treatment_col] == 1][ps_col]
    control_ps = df[df[treatment_col] == 0][ps_col]
    
    axes[0].hist(treated_ps, bins=30, alpha=0.6, color=COLORS['treated'], 
                 label='Treated', edgecolor='black')
    axes[0].hist(control_ps, bins=30, alpha=0.6, color=COLORS['control'], 
                 label='Control', edgecolor='black')
    axes[0].set_xlabel('Propensity Score', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Propensity Score Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Density plot
    treated_ps.plot(kind='density', ax=axes[1], color=COLORS['treated'], 
                    linewidth=2.5, label='Treated')
    control_ps.plot(kind='density', ax=axes[1], color=COLORS['control'], 
                    linewidth=2.5, label='Control')
    axes[1].set_xlabel('Propensity Score', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('Propensity Score Density', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    plt.show()


def plot_balance(balance_df: pd.DataFrame,
                title: Optional[str] = None,
                figsize: Tuple[int, int] = (10, 6),
                save_path: Optional[str] = None):
    """
    Plot covariate balance between treatment and control groups.
    
    Parameters:
    -----------
    balance_df : pd.DataFrame
        DataFrame with balance statistics (must have 'Std. Difference' column)
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    variables = balance_df['Variable'].values
    std_diffs = balance_df['Std. Difference'].values
    
    # Color based on balance threshold
    colors = [COLORS['success'] if abs(x) < 0.1 else COLORS['warning'] for x in std_diffs]
    
    # Horizontal bar plot
    y_pos = np.arange(len(variables))
    ax.barh(y_pos, std_diffs, color=colors, alpha=0.7, edgecolor='black')
    
    # Add reference lines
    ax.axvline(x=-0.1, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(x=0.1, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    
    # Add text annotation
    ax.text(0.11, len(variables) - 0.5, 'Imbalanced →', 
            fontsize=9, color='red', va='center')
    ax.text(-0.11, len(variables) - 0.5, '← Imbalanced', 
            fontsize=9, color='red', va='center', ha='right')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([v.replace('_', ' ').title() for v in variables])
    ax.set_xlabel('Standardized Difference', fontsize=12)
    ax.set_title(title or 'Covariate Balance Check', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame,
                           variables: List[str],
                           title: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 8),
                           save_path: Optional[str] = None):
    """
    Plot correlation matrix heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with variables
    variables : List[str]
        Variables to include in correlation matrix
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate correlation matrix
    corr = df[variables].corr()
    
    # Create heatmap
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)
    
    # Customize labels
    labels = [v.replace('_', ' ').title() for v in variables]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    plt.show()


def plot_scatter_with_regression(df: pd.DataFrame,
                                 x: str,
                                 y: str,
                                 hue: Optional[str] = None,
                                 title: Optional[str] = None,
                                 figsize: Tuple[int, int] = (10, 6),
                                 save_path: Optional[str] = None):
    """
    Plot scatter plot with regression line.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe
    x : str
        X variable
    y : str
        Y variable
    hue : str, optional
        Grouping variable for colors
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if hue:
        sns.scatterplot(data=df, x=x, y=y, hue=hue, s=100, alpha=0.6, ax=ax)
        
        # Add regression lines for each group
        for group in df[hue].unique():
            group_data = df[df[hue] == group]
            z = np.polyfit(group_data[x].dropna(), group_data[y].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(group_data[x].min(), group_data[x].max(), 100)
            ax.plot(x_line, p(x_line), linewidth=2, label=f'{hue}={group} (trend)')
    else:
        sns.scatterplot(data=df, x=x, y=y, s=100, alpha=0.6, color=COLORS['primary'], ax=ax)
        
        # Add regression line
        z = np.polyfit(df[x].dropna(), df[y].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[x].min(), df[x].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
    
    ax.set_xlabel(x.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y.replace('_', ' ').title(), fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    plt.show()


def plot_did_visualization(df: pd.DataFrame,
                          outcome: str,
                          treatment_col: str,
                          time_col: str,
                          treatment_time: int,
                          title: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 6),
                          save_path: Optional[str] = None):
    """
    Create comprehensive DiD visualization showing actual and counterfactual trends.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Panel dataframe
    outcome : str
        Outcome variable
    treatment_col : str
        Treatment indicator
    time_col : str
        Time variable
    treatment_time : int
        Time of treatment
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate means
    treated_avg = df[df[treatment_col] == 1].groupby(time_col)[outcome].mean()
    control_avg = df[df[treatment_col] == 0].groupby(time_col)[outcome].mean()
    
    # Split into pre and post
    pre_times = [t for t in treated_avg.index if t < treatment_time]
    post_times = [t for t in treated_avg.index if t >= treatment_time]
    
    # Plot actual trends
    ax.plot(treated_avg.loc[pre_times].index, treated_avg.loc[pre_times].values,
            'o-', linewidth=3, color=COLORS['treated'], label='Treated (Pre)', markersize=8)
    ax.plot(treated_avg.loc[post_times].index, treated_avg.loc[post_times].values,
            'o-', linewidth=3, color=COLORS['treated'], label='Treated (Post)', markersize=8)
    
    ax.plot(control_avg.index, control_avg.values,
            's-', linewidth=3, color=COLORS['control'], label='Control', markersize=8)
    
    # Add counterfactual (extrapolate pre-treatment trend)
    if len(pre_times) >= 2:
        pre_treated = treated_avg.loc[pre_times]
        z = np.polyfit(pre_times, pre_treated.values, 1)
        p = np.poly1d(z)
        counterfactual = p(post_times)
        ax.plot(post_times, counterfactual, '--', linewidth=2.5, 
                color=COLORS['treated'], alpha=0.5, label='Counterfactual')
    
    # Add vertical line at treatment
    ax.axvline(x=treatment_time, color='gray', linestyle='--', 
               linewidth=2, alpha=0.7, label='Treatment')
    
    ax.set_xlabel(time_col.capitalize(), fontsize=12)
    ax.set_ylabel(outcome.replace('_', ' ').title(), fontsize=12)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title('Difference-in-Differences Visualization', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    plt.show()



