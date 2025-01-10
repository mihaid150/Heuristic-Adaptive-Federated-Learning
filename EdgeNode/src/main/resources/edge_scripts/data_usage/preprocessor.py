import pandas as pd
import numpy as np
import os
import json


def extract_date_features(dataframe, date_column):
    """
    Extracts date-related features from a specified date column in a DataFrame.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The input DataFrame containing the date column.
    date_column : str
        The name of the column containing date values.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with additional date-related features:
        - year, month, day, weekday
        - sinusoidal representations of month and weekday.
    """
    dataframe[date_column] = pd.to_datetime(dataframe[date_column])
    dataframe['year'] = dataframe[date_column].dt.year
    dataframe['month'] = dataframe[date_column].dt.month
    dataframe['day'] = dataframe[date_column].dt.day
    dataframe['weekday'] = dataframe[date_column].dt.weekday
    dataframe['month_sin'] = np.sin((dataframe['month'] - 1) * (2. * np.pi / 12))
    dataframe['month_cos'] = np.cos((dataframe['month'] - 1) * (2. * np.pi / 12))
    dataframe['weekday_sin'] = np.sin((dataframe['weekday'] - 1) * (2. * np.pi / 7))
    dataframe['weekday_cos'] = np.cos((dataframe['weekday'] - 1) * (2. * np.pi / 7))
    return dataframe


def extract_time_features(dataframe, datetime_column):
    """
    Extracts time-related features from a specified datetime column in a DataFrame.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The input DataFrame containing the datetime column.
    datetime_column : str
        The name of the column containing datetime values.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with additional time-related features:
        - hour, minute
        - sinusoidal representations of hour and minute.
    """
    dataframe[datetime_column] = pd.to_datetime(dataframe[datetime_column])
    dataframe['hour'] = dataframe[datetime_column].dt.hour
    dataframe['minute'] = dataframe[datetime_column].dt.minute
    dataframe['hour_sin'] = np.sin(dataframe['hour'] * (2. * np.pi / 24))
    dataframe['hour_cos'] = np.cos(dataframe['hour'] * (2. * np.pi / 24))
    dataframe['minute_sin'] = np.sin(dataframe['minute'] * (2. * np.pi / 60))
    dataframe['minute_cos'] = np.cos(dataframe['minute'] * (2. * np.pi / 60))
    return dataframe


def add_rolling_features(dataframe, target_column, windows):
    """
    Adds rolling window statistics to a target column in a DataFrame.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The input DataFrame containing the target column.
    target_column : str
        The name of the column to calculate rolling features for.
    windows : list of int
        List of window sizes for rolling calculations.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with added rolling features:
        - rolling mean, min, and max for each window size.
    """
    for window in windows:
        dataframe[f'{target_column}_rolling_mean_{window}'] = dataframe[target_column].rolling(
            window=window).mean().interpolate(method='linear').ffill().bfill()
        dataframe[f'{target_column}_rolling_min_{window}'] = dataframe[target_column].rolling(
            window=window).min().interpolate(method='linear').ffill().bfill()
        dataframe[f'{target_column}_rolling_max_{window}'] = dataframe[target_column].rolling(
            window=window).max().interpolate(method='linear').ffill().bfill()
    return dataframe


def check_file_content(dataframe_path):
    """
    Checks whether the file at the specified path exists and is not empty.

    Parameters:
    -----------
    dataframe_path : str
        Path to the file to be checked.

    Returns:
    --------
    bool
        True if the file exists and is not empty, False otherwise.
    """
    if os.path.exists(dataframe_path) and os.path.getsize(dataframe_path) > 0:
        return True
    else:
        return False


def apply_functions(dataframe, functions):
    """
    Sequentially applies a list of functions to a DataFrame.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The input DataFrame to transform.
    functions : list of callable
        A list of functions to apply to the DataFrame.

    Returns:
    --------
    pd.DataFrame
        The transformed DataFrame.
    """
    for func in functions:
        dataframe = func(dataframe)
    return dataframe


def preprocess_training_data(dataframe_path):
    """
    Preprocesses training data by applying various transformations and saves it in JSON format.

    Parameters:
    -----------
    dataframe_path : str
        Path to the input CSV or JSON file containing the training data.

    Returns:
    --------
    None
        The processed data is saved back to the same path in JSON format.

    Raises:
    -------
    ValueError:
        If the file is empty or has an unsupported file extension.
    Exception:
        If an error occurs during processing.
    """
    if not check_file_content(dataframe_path):
        return

    try:
        # Determine file extension
        file_extension = os.path.splitext(dataframe_path)[1].lower()

        # Load data based on file extension
        if file_extension == '.csv':
            dataframe = pd.read_csv(dataframe_path)
        elif file_extension == '.json':
            dataframe = pd.read_json(dataframe_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        if dataframe.empty:
            raise ValueError('Loaded data is empty!')

        functions = [
            lambda df: df.rename(columns={'KWH/hh (per half hour) ': 'value'}),
            lambda df: extract_date_features(df, 'datetime'),
            lambda df: extract_time_features(df, 'datetime'),
            lambda df: df.replace('Null', np.nan),  # Replace 'Null' with NaN
            lambda df: df.dropna(subset=['value']),  # Drop rows where 'value' is NaN
            lambda df: df.astype({'value': 'float'}),  # Ensure 'value' is float
            lambda df: add_rolling_features(df, 'value', windows=[3, 6])
        ]

        dataframe = apply_functions(dataframe, functions)

        dataframe['value'] = pd.to_numeric(dataframe['value'], errors='coerce')
        dataframe = dataframe[np.isfinite(dataframe['value'])]
        dataframe = dataframe[dataframe['value'] > 0]

        # Keep only the required columns
        required_columns = [
            'value',
            'value_rolling_mean_3',
            'value_rolling_max_3',
            'value_rolling_min_3',
            'value_rolling_mean_6',
            'value_rolling_max_6'
        ]

        dataframe = dataframe[required_columns]

        # Save the processed DataFrame back to JSON format
        dataframe.to_json(dataframe_path, orient='records', date_format='iso')

    except Exception as e:
        print(json.dumps({"error from process_training_data.py: ": str(e)}))
