import os
import tensorflow as tf

from data_selection import filter_data_by_interval_date, filter_data_by_day_date
from edge_resources_paths import EdgeResourcesPaths

# Setting the environment variable to suppress TensorFlow low-level logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_initial_lstm_model():
    """
    Create and compile an initial LSTM model with a specified number of features and sequence length.

    Returns:
    tf.keras.Model: Compiled LSTM model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(48, 6), dtype=tf.float32),
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def pre_train_edge_model(edge_model_file_path: str, start_date: str, end_date: str):
    # load periodical data
    filter_data_by_interval_date(EdgeResourcesPaths.INPUT_DATA_PATH, "lorem ipsum", start_date, end_date,
                                 EdgeResourcesPaths.FILTERED_DATA_PATH)