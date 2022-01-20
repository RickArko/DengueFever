import pandas as pd
import seaborn as sns
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


def tsplot(y, lags=None, title='', figsize=(14, 8)):
    '''Examine the patterns of ACF and PACF, along with the time series plot and histogram.
    
    Original source: https://tomaugspurger.github.io/modern-7-timeseries.html
    '''
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax   = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax  = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    
    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax


def plotly_forecast(y_train, y_val, forecast, title=None):
    """Plot forecast
    """
    mae = mean_absolute_error(y_val, forecast)
    if title:
        title = title + f" MAE: {mae:.2f}"
    else:
        title = f"MAE: {mae:.2f}"
    
    assert y_train.name == y_val.name, f'y_train and y_val need same name'
    assert (forecast.index == y_val.index).all(), f'forecast index should equal validation index'

    forecast.name = y_train.name
    dfobs = pd.DataFrame(pd.concat([y_train, y_val])).assign(Type='Actual')
    dffc = pd.DataFrame(forecast, columns=['total_cases']).assign(Type='Forecast')
    
    dff = pd.concat([dfobs, dffc])
    fig = px.line(dff, x=dff.index, y=y_train.name, color='Type').update_layout(title=title)
    fig.show()
    return fig
