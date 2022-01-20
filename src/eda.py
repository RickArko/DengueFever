from pathlib import Path
from matplotlib.pyplot import plot
import pandas as pd
import plotly.express as px

from pandas_profiling import ProfileReport
from process import load_data, add_lags

import matplotlib.pyplot as plt
from plotting import tsplot


def plotly_variable_by_city(df, dependent_var, save=False):
    """Plotly chart of dependent variable by city over time.
    """
    fig = px.line(df[['week_start_date', dependent_var, 'city']], color='city', x='week_start_date', y=dependent_var)
    fig.update_layout(title=f"{dependent_var} By City")
    if save:
        fig.write_html(Path("eda").joinpath(f"{dependent_var}_by_city.html"))
    fig.show()
    return


def plotly_correlation(df, save=False, title=None):
    """Plotly chart of correlation all variables
    """
    if title:
        title = f'Correlation Plot: {title}'
    else:
        title = 'Correlation Plot'

    fig = px.imshow(df.corr())
    fig.update_layout(title=title)
    if save:
        fig.write_html(f"eda\{title.replace(' ', '').replace(':', '_')}.html")
    fig.show()


if __name__ == '__main__':
    dfin = load_data()

    separate_by_city = False    # determined to be not too useful

    profile = ProfileReport(dfin, title="Pandas Profiling Report", explorative=True)
    profile.to_file(Path("eda").joinpath("raw_dengue_eda.html"))
    plotly_variable_by_city(dfin, 'total_cases', save=True)

    for city, dfcity in dfin.groupby('city'):
        dfcity = dfcity.fillna(method='ffill').drop(['year', 'weekofyear'], 1)
        plotly_correlation(dfcity, title=f'{city}', save=True)
        tsplot(dfcity['total_cases'], 20, title=f'Time-Series Plots for {city}')
        plt.savefig(f'eda/tsplot_{city}.png')

    for city, dfcity in dfin.groupby('city'):
        print(f'Begin City Specific EDA:\n{city} with shape: {dfcity.shape}')
        dflags = add_lags(dfcity, lags=20)
        lag_cols = [c for c in dflags.columns if c.startswith('lag')] + ['total_cases']
        
        # skip this in favor of pcf/acf plots
        # plotly_correlation(dflags[lag_cols], title=city+' Lags')

        if separate_by_city:
            profile = ProfileReport(dfcity[[c for c in dfcity.columns if c not in lag_cols]],
                                    title=f"Profiling Report {city}", explorative=True)
            profile.to_file(Path("eda").joinpath(f"raw_dengue_eda_{city}.html"))