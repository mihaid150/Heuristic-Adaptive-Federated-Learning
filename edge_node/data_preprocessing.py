import os
import pandas as pd
import numpy as np
from shared.utils import required_columns
from shared.logging_config import logger


def extract_date_features(dataframe):
    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'], errors='coerce')
    dataframe = dataframe.dropna(subset=['datetime'])
    dataframe['year'] = dataframe['datetime'].dt.year
    dataframe['month'] = dataframe['datetime'].dt.month
    dataframe['day'] = dataframe['datetime'].dt.day
    dataframe['weekday'] = dataframe['datetime'].dt.weekday
    dataframe['month_sin'] = np.sin((dataframe['month'] - 1) * (2. * np.pi / 12))
    dataframe['month_cos'] = np.cos((dataframe['month'] - 1) * (2. * np.pi / 12))
    dataframe['weekday_sin'] = np.sin((dataframe['weekday'] - 1) * (2. * np.pi / 7))
    dataframe['weekday_cos'] = np.cos((dataframe['weekday'] - 1) * (2. * np.pi / 7))
    return dataframe


def extract_time_features(dataframe):
    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'], errors='coerce')
    dataframe = dataframe.dropna(subset=['datetime'])
    dataframe['hour'] = dataframe['datetime'].dt.hour
    dataframe['minute'] = dataframe['datetime'].dt.minute
    dataframe['hour_sin'] = np.sin(dataframe['hour'] * (2. * np.pi / 24))
    dataframe['hour_cos'] = np.cos(dataframe['hour'] * (2. * np.pi / 24))
    dataframe['minute_sin'] = np.sin(dataframe['minute'] * (2. * np.pi / 60))
    dataframe['minute_cos'] = np.cos(dataframe['minute'] * (2. * np.pi / 60))
    return dataframe


def add_rolling_features(dataframe, windows):
    """
    Compute rolling mean for each window size.
    """
    for window in windows:
        dataframe[f'value_rolling_mean_{window}'] = (
            dataframe['value']
            .rolling(window=window)
            .mean()
            .interpolate(method='linear')
            .ffill()
            .bfill()
        )
    return dataframe


def add_advanced_features(dataframe, windows):
    """
    Adds advanced features:
      - First difference of the 'value'
      - Exponential moving averages (EMA) for each window
      - Rolling volatility (standard deviation) for each window
      - Drift flag based on whether the absolute first difference exceeds twice the rolling volatility (using window 6 as baseline)
    """
    dataframe['value_diff'] = dataframe['value'].diff().fillna(0)
    for window in windows:
        dataframe[f'value_ewm_{window}'] = dataframe['value'].ewm(span=window, adjust=False).mean()
        dataframe[f'value_volatility_{window}'] = (
            dataframe['value']
            .rolling(window=window)
            .std()
            .fillna(0)
        )
    # For drift detection, use a baseline window of 6 (this can be tuned)
    baseline_volatility = dataframe['value'].rolling(window=6).std().fillna(0)
    threshold = baseline_volatility * 2
    dataframe['drift_flag'] = (dataframe['value_diff'].abs() > threshold).astype(int)
    return dataframe


def preprocess_data(data_file_path, timedate_column_name, value_column_name):
    if not os.path.exists(data_file_path) or os.path.getsize(data_file_path) <= 0:
        raise ValueError("Data file does not exist or it is empty!")

    dataframe = pd.read_csv(
        data_file_path,
        low_memory=False,
        dtype={timedate_column_name: str}
    )

    functions = [
        lambda df: df.rename(columns={timedate_column_name: "datetime"}),
        lambda df: df.rename(columns={value_column_name: "value"}),
        extract_date_features,
        extract_time_features,
        lambda df: df.replace('Null', np.nan),
        lambda df: df.dropna(subset=['value']),
        lambda df: df.astype({'value': 'float'}),
        lambda df: add_rolling_features(df, windows=[3, 6, 12, 24]),
        lambda df: add_advanced_features(df, windows=[3, 6, 12, 24])
    ]

    for function in functions:
        dataframe = function(dataframe)

    dataframe = dataframe[required_columns]
    dataframe.to_csv(data_file_path, index=False)
    logger.info(f"Updated file saved at: {data_file_path}")
