import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from tqdm import tqdm

# statsmodels
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

# Custom Feature Processing
from process import load_data, fill_missing_by_last

# Endogenous VAR/ARIMAX/SARIMAX
# from statsmodels.tsa.api import VAR


###
### Split Data:
###

dfin = load_data()

# start with one city for now (sj)...
df = dfin[dfin.city == 'sj']
df = fill_missing_by_last(df)
df = df.set_index('week_start_date').resample('W-SUN').sum()


# train_test_split not suitable for time-series
cut_point = int(np.ceil(len(df) * .8))
X_train, X_val = df.drop('total_cases', 1).iloc[0:cut_point], df.drop('total_cases', 1).iloc[cut_point:,:]
y_train, y_val = df['total_cases'][0:cut_point], df['total_cases'][cut_point:]


###
### Simple Auto Regression:
###

def train_and_view_ar(lags, trend='n', save=False):
    """Train AutoRegression with specified Lags and view results
    """
    model_ar = AutoReg(y_train, lags, trend=trend).fit(cov_type='HC0')
    pred = model_ar.predict(start=len(y_train), end=len(y_train) + len(y_val) - 1)

    print(f'avg MAD: {np.abs(pred - y_val).mean()}')
    dfval = pd.DataFrame(y_val).assign(Prediction=lambda x: "Actual")
    dfpred = pd.DataFrame(pred).rename(columns={0: 'total_cases'}).assign(Prediction=lambda x: "Prediction")
    dfres = pd.concat([dfval, dfpred])
    f = px.line(dfres.reset_index(), x='index', y='total_cases', color='Prediction')
    f.update_layout(title=f'AR Model with {lags} lags (Predicted vs. Actual)')
    f.show()
    if save:
        f.write_html(Path("models").joinpath(f"ARModel.html"))


# Example of a very naive optimization

# # check for trend (appears none or constant is best - somewhat obvious)
# for trend in ['n', 'c', 't', 'ct']:
#     for lags in range(1, 300, 50):
#         model = AutoReg(y_train, lags, trend=trend).fit(cov_type='HC0')
#         pred = model.predict(start=len(y_train), end=len(y_train) + len(y_val) - 1)
#         print(f'Trend: {trend} AR terms: {lags} avgMAD: {np.abs(pred - y_val).mean():,.2f}')


results = list()
for lag in range(1, 350, 1):
    model = AutoReg(y_train, lag, trend='n').fit(cov_type='HC0')
    pred = model.predict(start=len(y_train), end=len(y_train) + len(y_val) - 1)

    res_dict = {'AR_periods': int(lag),
                # 'trend': trend,
                'MAD': np.abs(pred - y_val).mean(),
                'MD': (pred - y_val).mean()
            }
    results.append(res_dict)
    if lag % 10 == 0:
        print(f'AR Periods: {lag}, MAD: {np.abs(pred - y_val).mean():,.2f}')

dfr = pd.DataFrame(results)
optimal_ar = int(dfr.loc[np.argmin(dfr.MAD)]['AR_periods'])

print(f'Found optimal AR periods to be: {optimal_ar} resulting in MAD: {dfr.loc[optimal_ar].MAD:,.2f}')


train_and_view_ar(1)
train_and_view_ar(10)
train_and_view_ar(100)
train_and_view_ar(optimal_ar, save=True)
