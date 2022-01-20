import pathlib
import numpy as np
import pandas as pd


def load_data():
    """Read and Combine Features and Labels.
    """
    dff = pd.read_csv(pathlib.Path('data/dengue_features.csv'))
    dfy = pd.read_csv(pathlib.Path('data/dengue_labels.csv'))
    df = dff.merge(dfy, on=['city', 'year', 'weekofyear'])
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    return df


def process_data(df):
    """Process data for time-series modelling.
    """
    df['dt'] = pd.to_datetime(df['week_start_date'])
    df = df.set_index('dt')
    y = df[['total_cases']].resample('W-SUN').sum()
    dff = df.resample('W-SUN').sum()
    dff.index.freq = 'W-SUN'
    return y, dff


DATA_DICT = {
    "SATELLITE": ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw'],
    "NOAA_TEMP": ['station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm'],
    "NOAA_CLIMATE": ['reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k',
                     'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent',
                     'reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k']        
}


def fill_missing_by_type(df):
    """
    """
    for col in DATA_DICT["SATELLITE"]:
        df[col] = df[col].fillna(df[col].shift(1))

    for col in DATA_DICT["NOAA_TEMP"]:
        print(f'figure missing for {col}')
        df[col] = df[col].fillna(df[col].shift(1))


def fill_missing_by_last(df):
    """Fill Missing by last value (except lag columns)
    """
    cols = [col for col in df.columns if col != 'lag']
    df[cols] = df[cols].fillna(method='ffill')
    return df


def split_data(df, percent_holdout=.8):
    """Split DataFrame into Train/Test by holdout percent.

    Returns: (X_tr, X_val, y_tr, y_val)
    """
    cut_point = int(np.ceil(len(df) * percent_holdout))
    X_train, X_val = df.drop('total_cases', 1).iloc[0:cut_point], df.drop('total_cases', 1).iloc[cut_point:,:]
    y_train, y_val = df['total_cases'][0:cut_point], df['total_cases'][cut_point:]

    return X_train, X_val, y_train, y_val


def add_lags(df, var='total_cases', lags=10):
    """Add specified number of lagged variables for var.
    """
    for lag in range(lags):
        # print(f'Computing lag period {lag} for {var}')
        df[f'lag{lag}_{var}'] = df[var].shift(lag)
    return df


if __name__ == '__main__':
        
    dfin = load_data()
    for city, dfcity in dfin.groupby('city'):
        y, dff = process_data(dfcity)

