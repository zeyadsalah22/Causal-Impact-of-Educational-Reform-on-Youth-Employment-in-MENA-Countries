"""
Data processing utilities for the MENA education-employment causal analysis.
Covers panel data loading, missing-value handling, feature engineering,
treatment indicator construction, and covariate balance checks.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

def load_panel_data(filepath: str) -> pd.DataFrame:
    """
    Load panel data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame : Loaded panel data
    """
    df = pd.read_csv(filepath)
    
    if 'year' in df.columns:
        if df['year'].dtype == 'object':
            # Remove 'YR' prefix if present and convert to int
            df['year'] = df['year'].astype(str).str.replace('YR', '', regex=False).astype(int)
        else:
            df['year'] = df['year'].astype(int)
    
    print(f"✓ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def handle_missing_data(df: pd.DataFrame, 
                       strategy: str = 'forward_fill',
                       threshold: float = 0.5) -> pd.DataFrame:
    """
    Handle missing data in panel dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Strategy for handling missing data:
        - 'forward_fill': Forward fill within countries
        - 'interpolate': Linear interpolation within countries
        - 'drop': Drop rows with missing values in key variables
    threshold : float
        Maximum proportion of missing data to keep a variable
    
    Returns:
    --------
    pd.DataFrame : Cleaned dataframe
    """
    df_clean = df.copy()
    
    country_col = 'country_code' if 'country_code' in df.columns else 'country'
    
    # Drop variables exceeding the missing-data threshold
    missing_prop = df_clean.isnull().sum() / len(df_clean)
    vars_to_keep = missing_prop[missing_prop < threshold].index
    df_clean = df_clean[vars_to_keep]
    
    print(f"Dropped {len(df.columns) - len(vars_to_keep)} variables with >{threshold*100}% missing data")
    
    if strategy == 'forward_fill':
        df_clean = df_clean.groupby(country_col).fillna(method='ffill')
        
    elif strategy == 'interpolate':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean.groupby(country_col)[col].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
    
    return df_clean


def create_lags(df: pd.DataFrame, 
                variables: List[str], 
                lags: List[int] = [1]) -> pd.DataFrame:
    """
    Create lagged variables for panel data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Panel dataframe with country column and year columns
    variables : List[str]
        List of variables to lag
    lags : List[int]
        List of lag periods
    
    Returns:
    --------
    pd.DataFrame : Dataframe with lagged variables
    """
    df_with_lags = df.copy()
    
    country_col = 'country_code' if 'country_code' in df.columns else 'country'
    
    for var in variables:
        for lag in lags:
            lag_col_name = f"{var}_lag{lag}"
            df_with_lags[lag_col_name] = df_with_lags.groupby(country_col)[var].shift(lag)
    
    return df_with_lags


def create_differences(df: pd.DataFrame, 
                      variables: List[str]) -> pd.DataFrame:
    """
    Create first differences for panel data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Panel dataframe
    variables : List[str]
        Variables to difference
    
    Returns:
    --------
    pd.DataFrame : Dataframe with differenced variables
    """
    df_with_diffs = df.copy()
    
    country_col = 'country_code' if 'country_code' in df.columns else 'country'
    
    for var in variables:
        diff_col_name = f"{var}_diff"
        df_with_diffs[diff_col_name] = df_with_diffs.groupby(country_col)[var].diff()
    
    return df_with_diffs


def create_treatment_indicator(df: pd.DataFrame,
                               treatment_var: str,
                               threshold: float,
                               method: str = 'median_year') -> Tuple[pd.DataFrame, dict]:
    """
    Create treatment indicator for difference-in-differences analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Panel dataframe
    treatment_var : str
        Variable to base treatment on (e.g., 'tertiary_enrollment')
    threshold : float
        Threshold value for treatment
    method : str
        Method for determining treatment timing:
        - 'median_year': Countries above median in later period
        - 'threshold': Countries that crossed threshold
    
    Returns:
    --------
    Tuple[pd.DataFrame, dict] : 
        - Dataframe with treatment indicators
        - Dictionary with treatment information
    """
    df_treated = df.copy()
    
    country_col = 'country_code' if 'country_code' in df.columns else 'country'
    
    if method == 'median_year':
        median_by_year = df.groupby('year')[treatment_var].median()
        
        mid_year = int(df['year'].median())
        pre_period_median = median_by_year[median_by_year.index < mid_year].median()
        post_period_median = median_by_year[median_by_year.index >= mid_year].median()
        
        # Treated: countries with above-median enrollment in the post period
        post_period_data = df[df['year'] >= mid_year]
        treated_countries = post_period_data.groupby(country_col)[treatment_var].mean()
        treated_countries = treated_countries[treated_countries > post_period_median].index.tolist()
        
        df_treated['treated'] = df_treated[country_col].isin(treated_countries).astype(int)
        df_treated['post'] = (df_treated['year'] >= mid_year).astype(int)
        df_treated['treated_post'] = df_treated['treated'] * df_treated['post']
        
        treatment_info = {
            'method': method,
            'pre_period': (int(df['year'].min()), int(mid_year - 1)),
            'post_period': (int(mid_year), int(df['year'].max())),
            'treated_countries': treated_countries,
            'n_treated': len(treated_countries),
            'n_control': df[country_col].nunique() - len(treated_countries)
        }
    
    elif method == 'threshold':
        df_treated['above_threshold'] = (df_treated[treatment_var] > threshold).astype(int)
        
        treatment_years = {}
        for country in df_treated[country_col].unique():
            country_data = df_treated[df_treated[country_col] == country]
            crossing_years = country_data[country_data['above_threshold'] == 1]['year']
            if len(crossing_years) > 0:
                treatment_years[country] = crossing_years.min()
        
        df_treated['treatment_year'] = df_treated[country_col].map(treatment_years)
        df_treated['treated'] = df_treated[country_col].isin(treatment_years.keys()).astype(int)
        df_treated['post'] = (df_treated['year'] >= df_treated['treatment_year']).astype(int)
        df_treated['post'] = df_treated['post'].fillna(0).astype(int)
        df_treated['treated_post'] = df_treated['treated'] * df_treated['post']
        
        treatment_info = {
            'method': method,
            'threshold': threshold,
            'treatment_years': treatment_years,
            'treated_countries': list(treatment_years.keys()),
            'n_treated': len(treatment_years),
            'n_control': df[country_col].nunique() - len(treatment_years)
        }
    
    return df_treated, treatment_info


def calculate_summary_stats(df: pd.DataFrame, 
                           variables: List[str],
                           groupby: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate summary statistics for variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    variables : List[str]
        Variables to summarize
    groupby : str, optional
        Variable to group by
    
    Returns:
    --------
    pd.DataFrame : Summary statistics
    """
    if groupby:
        summary = df.groupby(groupby)[variables].describe().T
    else:
        summary = df[variables].describe().T
    
    return summary


def check_balance(df: pd.DataFrame,
                 covariates: List[str],
                 treatment_col: str = 'treated') -> pd.DataFrame:
    """
    Check balance of covariates between treatment and control groups.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Panel dataframe with treatment indicator
    covariates : List[str]
        Covariates to check balance for
    treatment_col : str
        Name of treatment column
    
    Returns:
    --------
    pd.DataFrame : Balance statistics
    """
    balance_stats = []
    
    for var in covariates:
        treated_mean = df[df[treatment_col] == 1][var].mean()
        control_mean = df[df[treatment_col] == 0][var].mean()
        treated_std = df[df[treatment_col] == 1][var].std()
        control_std = df[df[treatment_col] == 0][var].std()
        
        # Standardized difference
        pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
        std_diff = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        balance_stats.append({
            'Variable': var,
            'Treated Mean': treated_mean,
            'Control Mean': control_mean,
            'Difference': treated_mean - control_mean,
            'Std. Difference': std_diff,
            'Balanced': abs(std_diff) < 0.1
        })
    
    return pd.DataFrame(balance_stats)



