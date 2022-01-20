# DengueFever
Forecast total cases of Dengue Fever in the cities of San Juan and Iquitos.

---

##  Checklist:
1. EDA\Visualizations
2. Analysis of Covariate Relation to Target
3. Documentation, Observations, Statistical Tests
4. Forecast Model

----

# Exploratory Data Analysis
Results saved in `src/eda`
```cd src
    python eda.py
```

# Baseline model (Simple AutoRegression)
Results saved in `src/ar_models`
```cd src
    python ar_model.py
```


# Modelling Thoughts

1. **Baseline** (something simple to improve on - lagged inputs only)
2. **Complex** (clearly exogeneous variables matter here, this is tricky though get to if time permits)

### Features/Processing (initial thoughts):
    1. Normalize (cases per 100k rather than total cases)
    2. Missing Data
        - mean
        - same as last
        - mode for categoric
        - MICE
    3. Standardize Timestamp Frequency
        - Not a big deal but make every period start on same day
        - Aggregate covariates appropriately
    3. Features
        - AutoRegressive (Lags)
            - Quickly test a few simple AR Models (lagged observations clearly matter)
        - Exogeneous
            - Clearly lagged variables don't contain all the information (non-cyclical plot)
            - This is a mosquito born virus and we're provided lots of NOAA data... 
                - Rainfaill/Humidity and Temperature likely important
                    - Linear Correlation is low (include plots)
                - Perhaps order matters?
                    - (i.e. mild winter followed by humid wet/hot season)
                    - El Nino? (perhaps more foreseeable compared to humidity/temp)

            - Including Exogeneous predictors is not easy in a time-series context
                - forecasts will be required for each co-variate, so it's important to establish a baseline and demonstrate meaningfull improvement


# Installation/Configuration
Need some version of `Python3.9` in `$PythonPath` then use `pipenv` to install dependencies:
```
    $PythonPath\python.exe -m pip install pipenv
    set PIPENV_VENV_IN_PROJECT="enabled"
    pipenv install
    pipenv shell
    cd src
```
