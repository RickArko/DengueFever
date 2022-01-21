from audioop import rms
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from sklearn.metrics import r2_score
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

from metrics import rmse, sMAPE
from plotting import plotly_forecast
from process import fill_missing_by_last, load_data, split_data

# Endogenous VAR/ARIMAX/SARIMAX
# from statsmodels.tsa.api import VAR


def plotly_train_test(y_train, y_val, title: str, save=False):
    """View train/test split.
    """
    dfsplit = pd.concat([pd.DataFrame(y_train).assign(set="train"),
                         pd.DataFrame(y_val).assign(set="test")])
    f = px.line(dfsplit, x=dfsplit.index, y='total_cases', color='set')
    f.update_layout(title=title)
    if save:
        f.write_html(f"eda/{title.replace(' ', '_')}.html")
    f.show()


def tune_ar(y_train, y_val, lags=300):
    """Tune optimal number of AR-lags.
    """
    results = list()
    for lag in tqdm(range(1, lags+1, 1)):
        # already determined no trend is optimal
        model = AutoReg(y_train, lag, trend='n').fit(
            cov_type='HC0')
        pred = model.predict(start=len(y_train),
                             end=len(y_train) + len(y_val) - 1)

        _mad, _rmse, _smape = np.abs(pred - y_val).mean(), rmse(pred, y_val), sMAPE(pred, y_val).mean()

        res_dict = {'AR_periods': int(lag),
                    'MAD': _mad,
                    'MD': (pred - y_val).mean(),
                    'RMSE': _rmse,
                    'sMAPE': _smape,
                    'R-square': r2_score(y_val, pred)
                    }
        results.append(res_dict)
        if lag % 10 == 0:
            print(f'Num: AR Periods: {lag}, MAD: {_mad:,.2f}, RMSE: {_rmse:,.2F}, sMAPE: {_smape}')

    dfr = pd.DataFrame(results)
    return dfr


def train_and_view_ar(y_train, y_val, lags, trend='n', save=False, title=None, verbose=False):
    """Train AutoRegression with specified Lags and view results
    """
    model_ar = AutoReg(y_train, lags, trend=trend).fit(cov_type='HC0')
    pred = model_ar.predict(start=len(y_train), end=len(y_train) + len(y_val) - 1)

    print(f'avg MAD: {np.abs(pred - y_val).mean()}')
    dfval = pd.DataFrame(y_val).assign(Prediction=lambda x: "Actual")
    dfpred = pd.DataFrame(pred).rename(columns={0: 'total_cases'}).assign(Prediction=lambda x: "Prediction")
    dfres = pd.concat([dfval, dfpred])
    f = px.line(dfres, x=dfres.index, y='total_cases', color='Prediction')
    f.update_layout(title=f'AR Model with {lags} lags (Predicted vs. Actual)')
    f.show()

    if verbose:
        print(model_ar.summary())

    if save:
        if title:
            title = f"ARModel_{title}.html"
        else:
            tile = f"ARModel.html"
        f.write_html(Path("ar_models").joinpath(title))



###
### Appears to be no trend, but ran a quick-dirty naive optimization to verify 'n' is best
###

# # check for trend (appears none or constant is best - somewhat obvious)
# for trend in ['n', 'c', 't', 'ct']:
#     for lags in range(1, 300, 50):
#         model = AutoReg(y_train, lags, trend=trend).fit(cov_type='HC0')
#         pred = model.predict(start=len(y_train), end=len(y_train) + len(y_val) - 1)
#         print(f'Trend: {trend} AR terms: {lags} avgMAD: {np.abs(pred - y_val).mean():,.2f}')




if __name__ == '__main__':
    dfin = load_data()

    holdout = .8
    for city, dfcity in dfin.groupby('city'):
        print(city, dfcity.shape)
        dfcity = fill_missing_by_last(dfcity)
        dfcity = dfcity.set_index('week_start_date').resample('W-SUN').sum()
        X_train, X_val, y_train, y_val = split_data(dfcity, percent_holdout=holdout)

        _title = f'Training Validation Split For {city} with {int(holdout * 100)}% holdout'
        print(_title, X_train.shape, X_val.shape, y_train.shape, y_val.shape)
        plotly_train_test(y_train, y_val, _title, save=True)

        if city == 'iq':
            lags = 200
        else:
            lags = 300
        dfresults = tune_ar(y_train, y_val, lags=lags)

        # Choose AR to max MAE/RMSE
        optimal_ar = int(dfresults.loc[np.argmin(dfresults.MAD)]['AR_periods'])
        optimal_ar_rmse = int(dfresults.loc[np.argmin(dfresults.RMSE)]['AR_periods'])
        optimal_ar_r2 = int(dfresults.loc[np.argmax(dfresults['R-square'])]['AR_periods'])
        optimal_ar_smape = int(dfresults.loc[np.argmin(dfresults['sMAPE'])]['AR_periods'])


        train_and_view_ar(y_train, y_val, optimal_ar_r2, save=True, title=F"Best_Rsquared_{city}")          # badly underfits, R-squared is a poor metric for fit here
        train_and_view_ar(y_train, y_val, optimal_ar_rmse, save=True, title=F"Best_RMSE_{city}")            # About same as MAE
        train_and_view_ar(y_train, y_val, optimal_ar, save=True, title=F"Best_MAE_{city}")                  # select this slightly more straight-forward
        train_and_view_ar(y_train, y_val, optimal_ar_smape, save=True, title=F"Best_sMAPE_{city}")          # select this slightly more straight-forward


        model_ar = AutoReg(y_train, optimal_ar, trend='n').fit(cov_type='HC0')
        pred = model_ar.predict(start=len(y_train), end=len(y_train) + len(y_val) - 1)
        fc = plotly_forecast(y_train, y_val, pred, title=f'Forcecast {city}')
        fc.write_html(f'ar_models/ARForecast_{city}.html')

    ### Simple AR model captures a good bit of the non-epidemic trend, but severely underpredicts during epidemic tops
    ### SJ:
        ### June 12, 2005 - Dec, 11 2005
        ### May 20, 2007 - Dec, 1 2007
        ### smells like el nino

    ### IQ:
        ### Sep, 2008 - Dec, 2008
        ### Jan 2009 - Mar, 2009
        ### Dec 2009 - May, 2010

