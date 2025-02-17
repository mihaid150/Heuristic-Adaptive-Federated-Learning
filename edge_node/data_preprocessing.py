import os
import pandas as pd
import numpy as np
from shared.logging_config import logger


def extract_date_features(dataframe):
    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])
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
    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])
    dataframe['hour'] = dataframe['datetime'].dt.hour
    dataframe['minute'] = dataframe['datetime'].dt.minute
    dataframe['hour_sin'] = np.sin(dataframe['hour'] * (2. * np.pi / 24))
    dataframe['hour_cos'] = np.cos(dataframe['hour'] * (2. * np.pi / 24))
    dataframe['minute_sin'] = np.sin(dataframe['minute'] * (2. * np.pi / 60))
    dataframe['minute_cos'] = np.cos(dataframe['minute'] * (2. * np.pi / 60))
    return dataframe


def add_rolling_features(dataframe, windows):
    for window in windows:
        dataframe[f'value_rolling_mean_{window}'] = dataframe['value'].rolling(
            window=window).mean().interpolate(method='linear').ffill().bfill()
        dataframe[f'value_rolling_min_{window}'] = dataframe['value'].rolling(
            window=window).min().interpolate(method='linear').ffill().bfill()
        dataframe[f'value_rolling_max_{window}'] = dataframe['value'].rolling(
            window=window).max().interpolate(method='linear').ffill().bfill()
    return dataframe


def preprocess_data(data_file_path, timedate_column_name, value_column_name):
    if not os.path.exists(data_file_path) or not os.path.getsize(data_file_path) > 0:
        raise ValueError("Data file does not exist or it is empty!")

    dataframe = pd.read_csv(data_file_path)

    functions = [
        lambda df: df.rename(columns={timedate_column_name: "datetime"}),
        lambda df: df.rename(columns={value_column_name: "value"}),
        lambda df: extract_date_features(df),
        lambda df: extract_time_features(df),
        lambda df: df.replace('Null', np.nan),
        lambda df: df.dropna(subset=['value']),
        lambda df: df.astype({'value': 'float'}),
        lambda df: add_rolling_features(df, windows=[3, 6])
    ]

    for function in functions:
        dataframe = function(dataframe)

    required_columns = [
        'value',
        'value_rolling_mean_3',
        'value_rolling_max_3',
        'value_rolling_min_3',
        'value_rolling_mean_6',
        'value_rolling_max_6'
    ]

    dataframe = dataframe[required_columns]
    dataframe.to_csv(data_file_path, index=False)
    logger.info(f"Updated file saved at: {data_file_path}")
