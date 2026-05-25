"""
Causal inference methods for MENA panel data analysis.
Implements Difference-in-Differences, Propensity Score Matching,
Event Study, and placebo testing.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DifferenceInDifferences:
    """
    Two-way fixed effects DiD estimator.

    Model: Y_it = β0 + β1·Treated_i + β2·Post_t + β3·Treated_i·Post_t + ε_it
    β3 is the ATT under the parallel trends assumption.
    """
    
    def __init__(self, df: pd.DataFrame, outcome: str, 
                 treatment_col: str = 'treated',
                 post_col: str = 'post',
                 controls: Optional[List[str]] = None):
        """
        Initialize DiD estimator.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Panel dataframe with treatment indicators
        outcome : str
            Outcome variable name
        treatment_col : str
            Treatment group indicator (0/1)
        post_col : str
            Post-treatment period indicator (0/1)
        controls : List[str], optional
            Control variables to include
        """
        self.df = df.copy()
        self.outcome = outcome
        self.treatment_col = treatment_col
        self.post_col = post_col
        self.controls = controls or []
        self.results = None
        self.country_col = 'country_code' if 'country_code' in df.columns else 'country'
        
    def estimate(self, fixed_effects: str = 'none') -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Estimate DiD model.
        
        Parameters:
        -----------
        fixed_effects : str
            Type of fixed effects: 'none', 'country', 'year', 'twoway'
        
        Returns:
        --------
        Regression results object
        """
        # Create interaction term
        self.df['treated_post'] = self.df[self.treatment_col] * self.df[self.post_col]
        
        # Build formula
        if fixed_effects == 'none':
            formula = f"{self.outcome} ~ {self.treatment_col} + {self.post_col} + treated_post"
            if self.controls:
                formula += " + " + " + ".join(self.controls)
                
        elif fixed_effects == 'country':
            formula = f"{self.outcome} ~ {self.post_col} + treated_post + C({self.country_col})"
            if self.controls:
                formula += " + " + " + ".join(self.controls)
                
        elif fixed_effects == 'year':
            formula = f"{self.outcome} ~ {self.treatment_col} + treated_post + C(year)"
            if self.controls:
                formula += " + " + " + ".join(self.controls)
                
        elif fixed_effects == 'twoway':
            formula = f"{self.outcome} ~ treated_post + C({self.country_col}) + C(year)"
            if self.controls:
                formula += " + " + " + ".join(self.controls)
        
        self.results = smf.ols(formula, data=self.df).fit(cov_type='cluster', 
                                                           cov_kwds={'groups': self.df[self.country_col]})
        
        return self.results
    
    def get_att(self) -> Dict:
        """
        Get Average Treatment Effect on the Treated (ATT).
        
        Returns:
        --------
        Dict with ATT, standard error, p-value, and confidence interval
        """
        if self.results is None:
            raise ValueError("Must run estimate() first")
        
        att = self.results.params['treated_post']
        se = self.results.bse['treated_post']
        pval = self.results.pvalues['treated_post']
        ci = self.results.conf_int().loc['treated_post']
        
        return {
            'ATT': att,
            'Std. Error': se,
            'p-value': pval,
            'CI_lower': ci[0],
            'CI_upper': ci[1],
            'Significant': pval < 0.05
        }
    
    def parallel_trends_test(self, pre_periods: int = 3) -> pd.DataFrame:
        """
        Test parallel trends assumption using pre-treatment periods.
        
        Parameters:
        -----------
        pre_periods : int
            Number of pre-treatment periods to test
        
        Returns:
        --------
        pd.DataFrame : Test results for each period
        """
        pre_data = self.df[self.df[self.post_col] == 0].copy()
        
        years = sorted(pre_data['year'].unique())
        last_years = years[-pre_periods:]
        
        test_results = []
        for year in last_years:
            pre_data['year_dummy'] = (pre_data['year'] == year).astype(int)
            pre_data['treated_year'] = pre_data[self.treatment_col] * pre_data['year_dummy']
            
            formula = f"{self.outcome} ~ {self.treatment_col} + year_dummy + treated_year"
            model = smf.ols(formula, data=pre_data).fit()
            
            test_results.append({
                'Year': year,
                'Coefficient': model.params.get('treated_year', 0),
                'Std. Error': model.bse.get('treated_year', 0),
                'p-value': model.pvalues.get('treated_year', 1),
                'Significant': model.pvalues.get('treated_year', 1) < 0.05
            })
        
        return pd.DataFrame(test_results)


class PropensityScoreMatching:
    """
    Nearest-neighbor propensity score matching estimator.
    Propensity scores are estimated via logistic regression;
    ATT is computed on the matched sample.
    """
    
    def __init__(self, df: pd.DataFrame, 
                 treatment_col: str,
                 outcome: str,
                 covariates: List[str]):
        """
        Initialize PSM estimator.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with treatment and covariates
        treatment_col : str
            Treatment variable name
        outcome : str
            Outcome variable name
        covariates : List[str]
            Covariates for propensity score model
        """
        self.df = df.copy().dropna(subset=[treatment_col, outcome] + covariates)
        self.treatment_col = treatment_col
        self.outcome = outcome
        self.covariates = covariates
        self.propensity_scores = None
        self.matched_data = None
        
    def estimate_propensity_scores(self) -> np.ndarray:
        """
        Estimate propensity scores using logistic regression.
        
        Returns:
        --------
        np.ndarray : Propensity scores
        """
        X = self.df[self.covariates]
        y = self.df[self.treatment_col]
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        self.propensity_scores = model.predict_proba(X)[:, 1]
        self.df['propensity_score'] = self.propensity_scores
        
        return self.propensity_scores
    
    def match(self, method: str = 'nearest', n_neighbors: int = 1, 
              caliper: float = 0.1) -> pd.DataFrame:
        """
        Match treated and control units based on propensity scores.
        
        Parameters:
        -----------
        method : str
            Matching method: 'nearest' or 'caliper'
        n_neighbors : int
            Number of neighbors to match
        caliper : float
            Maximum allowed difference in propensity scores
        
        Returns:
        --------
        pd.DataFrame : Matched dataset
        """
        if self.propensity_scores is None:
            self.estimate_propensity_scores()
        
        treated = self.df[self.df[self.treatment_col] == 1]
        control = self.df[self.df[self.treatment_col] == 0]
        
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        nn.fit(control[['propensity_score']])
        
        distances, indices = nn.kneighbors(treated[['propensity_score']])
        
        # Apply caliper if specified
        matched_control_indices = []
        matched_treated_indices = []
        
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if method == 'caliper' and dist[0] > caliper:
                continue
            matched_treated_indices.append(treated.index[i])
            matched_control_indices.extend(control.iloc[idx].index.tolist())
        
        matched_treated = self.df.loc[matched_treated_indices].copy()
        matched_control = self.df.loc[matched_control_indices].copy()
        
        self.matched_data = pd.concat([matched_treated, matched_control])
        
        print(f"✓ Matching complete:")
        print(f"  Treated units: {len(matched_treated)}")
        print(f"  Control units: {len(matched_control)}")
        print(f"  Total matched: {len(self.matched_data)}")
        
        return self.matched_data
    
    def estimate_att(self) -> Dict:
        """
        Estimate Average Treatment Effect on Treated (ATT) using matched sample.
        
        Returns:
        --------
        Dict with ATT and statistics
        """
        if self.matched_data is None:
            raise ValueError("Must run match() first")
        
        treated_outcome = self.matched_data[self.matched_data[self.treatment_col] == 1][self.outcome].mean()
        control_outcome = self.matched_data[self.matched_data[self.treatment_col] == 0][self.outcome].mean()
        
        att = treated_outcome - control_outcome
        
        treated_var = self.matched_data[self.matched_data[self.treatment_col] == 1][self.outcome].var()
        control_var = self.matched_data[self.matched_data[self.treatment_col] == 0][self.outcome].var()
        n_treated = (self.matched_data[self.treatment_col] == 1).sum()
        n_control = (self.matched_data[self.treatment_col] == 0).sum()
        
        se = np.sqrt(treated_var / n_treated + control_var / n_control)
        
        t_stat = att / se if se > 0 else 0
        from scipy import stats
        pval = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_treated + n_control - 2))
        
        return {
            'ATT': att,
            'Std. Error': se,
            't-statistic': t_stat,
            'p-value': pval,
            'Treated Mean': treated_outcome,
            'Control Mean': control_outcome,
            'N Treated': n_treated,
            'N Control': n_control
        }
    
    def check_balance(self) -> pd.DataFrame:
        """
        Check covariate balance after matching.
        
        Returns:
        --------
        pd.DataFrame : Balance statistics
        """
        if self.matched_data is None:
            raise ValueError("Must run match() first")
        
        balance_stats = []
        
        for var in self.covariates:
            treated_mean = self.matched_data[self.matched_data[self.treatment_col] == 1][var].mean()
            control_mean = self.matched_data[self.matched_data[self.treatment_col] == 0][var].mean()
            treated_std = self.matched_data[self.matched_data[self.treatment_col] == 1][var].std()
            control_std = self.matched_data[self.matched_data[self.treatment_col] == 0][var].std()
            
            pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
            std_diff = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0
            
            balance_stats.append({
                'Variable': var,
                'Treated Mean': treated_mean,
                'Control Mean': control_mean,
                'Std. Difference': std_diff,
                'Balanced': abs(std_diff) < 0.1
            })
        
        return pd.DataFrame(balance_stats)


class EventStudy:
    """
    Event study estimator for dynamic treatment effects.
    Estimates coefficients for each period relative to treatment onset,
    providing a visual test of pre-trends and post-treatment dynamics.
    """
    
    def __init__(self, df: pd.DataFrame, 
                 outcome: str,
                 treatment_col: str,
                 time_col: str = 'year',
                 treatment_year_col: str = 'treatment_year'):
        """
        Initialize Event Study.
        
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
        treatment_year_col : str
            Year of treatment for each unit
        """
        self.df = df.copy()
        self.outcome = outcome
        self.treatment_col = treatment_col
        self.time_col = time_col
        self.treatment_year_col = treatment_year_col
        self.results = None
        self.country_col = 'country_code' if 'country_code' in df.columns else 'country'
        
    def create_event_time(self, reference_period: int = -1):
        """
        Create event time variable (years relative to treatment).
        
        Parameters:
        -----------
        reference_period : int
            Reference period (usually -1, the period before treatment)
        """
        self.df['event_time'] = self.df[self.time_col] - self.df[self.treatment_year_col]
        self.df['event_time'] = self.df['event_time'].fillna(0)
        
        event_times = sorted(self.df['event_time'].unique())
        for t in event_times:
            if t != reference_period:
                self.df[f'event_time_{int(t)}'] = (self.df['event_time'] == t).astype(int)
    
    def estimate(self, controls: Optional[List[str]] = None) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Estimate event study model.
        
        Parameters:
        -----------
        controls : List[str], optional
            Control variables
        
        Returns:
        --------
        Regression results
        """
        self.create_event_time()
        
        event_time_vars = [col for col in self.df.columns if col.startswith('event_time_')]
        formula = f"{self.outcome} ~ " + " + ".join(event_time_vars)
        
        if controls:
            formula += " + " + " + ".join(controls)
        
        formula += f" + C({self.country_col}) + C(year)"
        
        self.results = smf.ols(formula, data=self.df).fit(cov_type='cluster',
                                                           cov_kwds={'groups': self.df[self.country_col]})
        
        return self.results
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Extract event study coefficients.
        
        Returns:
        --------
        pd.DataFrame : Coefficients by event time
        """
        if self.results is None:
            raise ValueError("Must run estimate() first")
        
        coeffs = []
        for param in self.results.params.index:
            if param.startswith('event_time_'):
                event_time = int(param.split('_')[-1])
                coeffs.append({
                    'Event Time': event_time,
                    'Coefficient': self.results.params[param],
                    'Std. Error': self.results.bse[param],
                    'CI_lower': self.results.conf_int().loc[param, 0],
                    'CI_upper': self.results.conf_int().loc[param, 1]
                })
        
        return pd.DataFrame(coeffs).sort_values('Event Time')


def placebo_test(df: pd.DataFrame,
                outcome: str,
                treatment_col: str,
                post_col: str,
                fake_treatment_year: int,
                controls: Optional[List[str]] = None) -> Dict:
    """
    Conduct placebo test with fake treatment timing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Panel dataframe
    outcome : str
        Outcome variable
    treatment_col : str
        Treatment indicator
    post_col : str
        Post period indicator
    fake_treatment_year : int
        Fake treatment year for placebo
    controls : List[str], optional
        Control variables
    
    Returns:
    --------
    Dict with placebo test results
    """
    df_placebo = df.copy()
    df_placebo['fake_post'] = (df_placebo['year'] >= fake_treatment_year).astype(int)
    df_placebo['fake_treated_post'] = df_placebo[treatment_col] * df_placebo['fake_post']
    
    formula = f"{outcome} ~ {treatment_col} + fake_post + fake_treated_post"
    if controls:
        formula += " + " + " + ".join(controls)
    
    results = smf.ols(formula, data=df_placebo).fit()
    
    placebo_effect = results.params.get('fake_treated_post', 0)
    placebo_pval = results.pvalues.get('fake_treated_post', 1)
    
    return {
        'Fake Treatment Year': fake_treatment_year,
        'Placebo Effect': placebo_effect,
        'p-value': placebo_pval,
        'Significant': placebo_pval < 0.05,
        'Pass Test': placebo_pval >= 0.05
    }



