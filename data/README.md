# Data Documentation

## Data Sources

### 1. World Bank Education Statistics
- **Indicator**: `SE.TER.ENRR` - Tertiary education enrollment (% gross)
- **Coverage**: MENA countries, 1990-2023
- **Source**: World Bank Open Data API

### 2. ILO Youth Unemployment
- **Indicator**: `SL.UEM.1524.ZS` - Unemployment, youth total (% of total labor force ages 15-24)
- **Coverage**: MENA countries, 1990-2023
- **Source**: World Bank via ILO data

### 3. Control Variables
- GDP per capita (constant 2015 US$): `NY.GDP.PCAP.KD`
- Population growth: `SP.POP.GROW`
- Government expenditure on education (% of GDP): `SE.XPD.TOTL.GD.ZS`
- Urban population (% of total): `SP.URB.TOTL.IN.ZS`

## MENA Countries Included

1. Algeria
2. Bahrain
3. Djibouti
4. Egypt, Arab Rep.
5. Iran, Islamic Rep.
6. Iraq
7. Jordan
8. Kuwait
9. Lebanon
10. Libya
11. Morocco
12. Oman
13. Qatar
14. Saudi Arabia
15. Syrian Arab Republic
16. Tunisia
17. United Arab Emirates
18. West Bank and Gaza
19. Yemen, Rep.

## Data Processing Steps

1. **Download**: Retrieved via World Bank API using `wbgapi` library
2. **Cleaning**: Handle missing values, standardize country names
3. **Merging**: Combine all indicators into panel dataset
4. **Transformation**: Create treatment indicators, lag variables
5. **Output**: Clean CSV files in `processed/` folder

## Files

- `raw/`: Original downloaded data (not tracked in git)
- `processed/`: Cleaned and merged datasets ready for analysis
  - `mena_panel.csv`: Main panel dataset
  - `mena_summary.csv`: Descriptive statistics by country

## Notes

- Some countries have missing data for certain years
- Treatment timing varies by country (educational reforms)
- All monetary values in constant 2015 US dollars

