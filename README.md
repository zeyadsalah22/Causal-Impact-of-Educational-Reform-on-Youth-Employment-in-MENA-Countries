# Causal Impact of Educational Reform on Youth Employment in MENA Countries

## Research Question

**Does increasing tertiary education enrollment causally reduce youth unemployment rates in MENA countries, or does it lead to credential inflation and unemployment of educated youth?**

## Overview

This project investigates the causal relationship between tertiary education expansion and youth employment outcomes in Middle East and North Africa (MENA) countries. The region presents a striking paradox: while education levels have risen dramatically over the past three decades, youth unemployment remains among the highest globally (~30%). Separating correlation from causation in this context requires rigorous quasi-experimental methods applied to panel data.

The analysis implements difference-in-differences, propensity score matching, and event study methods using publicly available World Bank and ILO data across 19 MENA countries from 1990 to 2023.

## Data Sources

- **World Bank Education Statistics**: Tertiary enrollment rates (1990–2023)
- **ILO Labor Statistics**: Youth unemployment rates by education level
- **World Bank Development Indicators**: GDP, population, economic controls
- **Coverage**: 19 MENA countries, 30+ years

## Methodology

### Causal Identification Strategy

1. **Directed Acyclic Graphs (DAGs)**: Mapping confounders and causal pathways
2. **Difference-in-Differences (DiD)**: Comparing countries with education reforms against control countries before and after treatment
3. **Propensity Score Matching (PSM)**: Constructing balanced treatment/control groups on observable covariates
4. **Event Study Analysis**: Examining dynamic treatment effects over the pre- and post-reform horizon

### Robustness Checks

- Parallel trends assumption testing using pre-treatment periods
- Placebo tests with falsified treatment timing
- Alternative control group specifications
- Sensitivity to unobserved confounders

## Project Structure

```
.
├── data/
│   ├── raw/              # Original datasets from APIs
│   ├── processed/        # Cleaned and merged data
│   └── README.md         # Data documentation
├── notebooks/
│   ├── 01_data_collection.ipynb       # Data gathering via World Bank API
│   ├── 02_exploratory_analysis.ipynb  # EDA and descriptive statistics
│   ├── 03_causal_identification.ipynb # Treatment definition and identification
│   ├── 04_estimation.ipynb            # Main causal estimation
│   └── 05_robustness_checks.ipynb     # Validation and sensitivity analysis
├── src/
│   ├── data_processing.py    # Data cleaning utilities
│   ├── causal_methods.py     # Causal inference implementations
│   └── visualization.py      # Plotting functions
├── results/
│   ├── figures/              # Generated plots
│   └── tables/               # Regression tables and summaries
├── requirements.txt          # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Running the Analysis
Execute notebooks in order:
1. `01_data_collection.ipynb` — downloads data from the World Bank API
2. `02_exploratory_analysis.ipynb` — descriptive statistics and trend analysis
3. `03_causal_identification.ipynb` — treatment definition and covariate balance
4. `04_estimation.ipynb` — DiD and PSM estimation
5. `05_robustness_checks.ipynb` — parallel trends, placebo tests, sensitivity

## Results

### Difference-in-Differences

Countries with above-median tertiary education expansion experienced approximately **5.2 percentage points lower youth unemployment** relative to control countries (p = 0.081). The two-way fixed effects model achieves R² = 0.907, absorbing substantial country- and year-level heterogeneity.

### Propensity Score Matching

The PSM ATT estimate is −2.2 percentage points (p = 0.315). The matching procedure balanced 109 treated observations against 109 controls, achieving standardized mean differences below 0.1 across all covariates.

### Interpretation

Both methods point toward a modest negative effect of tertiary education expansion on youth unemployment, though the PSM estimate does not reach conventional significance thresholds, likely reflecting limited statistical power at the country level. The DiD estimate is more credible given the panel structure and two-way fixed effects. Together, the findings suggest that education expansion may reduce youth unemployment in MENA, but that education policy alone is insufficient to resolve the region's structural employment challenge.

### Robustness

Pre-treatment parallel trends tests show no significant divergence between treated and control groups prior to the reform period. Placebo tests with falsified treatment dates produce no significant effects, supporting the validity of the identification strategy.

## Technologies

- **Python 3.8+**
- **Pandas / NumPy**: Data manipulation and computation
- **Statsmodels**: OLS with clustered standard errors, panel estimation
- **scikit-learn**: Propensity score estimation via logistic regression
- **DoWhy**: Causal inference framework
- **Matplotlib / Seaborn**: Visualization
- **wbgapi**: World Bank API client

## References

- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Morgan, S. L., & Winship, C. (2015). *Counterfactuals and Causal Inference*. Cambridge University Press.
- World Bank. (2020). *MENA Economic Update*.
- ILO. (2023). *Global Employment Trends for Youth*.

## Author

Zeyad Salah

## License

This project is open source and available for academic use.
