from enum import auto
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn import ensemble

# statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from plotting import plotly_forecast

# Custom Feature Processing
from process import load_data, fill_missing_by_last, split_data

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# ARIMA wants stationary data
def test_stationary(ts):
    """Test if timeseries (ts) is stationary with Augmented Dicky Fuller.
    """
    adfuller(ts, autolag='AIC')
    adf = pd.Series(adfuller(ts, autolag='AIC'), index=['adf', 'p-value', 'usedlag', 'nobs', 'criticalvalues', 'icbest'])
    if adf['p-value'] < .01:
        print('Time Series is Strongly Stationary')
    return adf


# holdout = .8
# dfcity = dfin[dfin['city'] == 'sj']
# # Example n-step ahead forecast
# y = dfcity['total_cases']
# size = int(len(y) * holdout)
# y, y_val = y[0:size], y[size:len(y)]

# steps = 12
# forecasts = np.ceil(len(y_val) / steps)



# for time in np.array_split(y_val.index, forecasts):
#     _start = time[0] - pd.Timedelta(days=1)
#     _end = time[-1]

#     y_tr = y.loc[: _start]
#     y_tst = y.loc[time[0]: time[-1] + pd.Timedelta(days=8)]

#     print(f'Forecast {steps} weeks from {time[0].date()} thru {_end.date()} training with {len(y_tr)} weeks before {y_tr.index.max().date()}')

#     # aa = auto_arima(y_tr)
#     # preds = aa.predict(steps)

#     model = ARIMA(y_train, order=(6,1,0))
#     model_fit = model.fit()
#     preds = model_fit.forecast(len(y_tst))

#     model_ar = AutoReg(y_train, 10, trend='n').fit(cov_type='HC0')
#     preds = model_ar.predict(start=len(y_tr), end=len(y_tr) + steps - 1)

#     plotly_forecast(y_tr, y_tst, preds)
#     preds = aa.fo
#     print(f'')
#     y.loc[:time]

#     for week in range(len(y_val)):
#         model = ARIMA(y_train) 



if __name__ == '__main__':

    dfin = load_data()

    # test if stationary
    for city, dfcity in dfin.groupby('city'):
        print(f'Teting if {city} time-series is stationary')
        adf = test_stationary(dfcity['total_cases'])
        print(adf)


    holdout = .8
    for city, dfcity in dfin.groupby('city'):
        print(city, dfcity.shape)
        dfcity = fill_missing_by_last(dfcity)
        dfcity = dfcity.set_index('week_start_date').resample('W-SUN').sum()
        X_train, X_val, y_train, y_val = split_data(dfcity, percent_holdout=holdout)


        model = ARIMA(y_train, order=(3,1,0))
        model_fit = model.fit()
        preds = model_fit.forecast(len(y_val))
        print(f'Basic ARIMA: {np.abs(preds - y_val).mean():.2f}')

