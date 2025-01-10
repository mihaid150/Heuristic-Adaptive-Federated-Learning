import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import traceback
from data_usage.preprocessor import preprocess_training_data
import logging
import warnings
import joblib

logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def add_synthetic_rows(df, target_length):
    """
    Adds synthetic rows to a DataFrame to reach the target length.

    This function duplicates the DataFrame and adds small random noise to numeric columns.

    Args:
    -----
    df : pd.DataFrame
        Input DataFrame.
    target_length : int
        Desired number of rows in the DataFrame.

    Returns:
    --------
    pd.DataFrame
        DataFrame with added synthetic rows.
    """
    while len(df) < target_length:
        synthetic_data = df.sample(frac=1, replace=True, random_state=42).copy()
        for col in synthetic_data.select_dtypes(include=[np.number]).columns:
            synthetic_data[col] += np.random.normal(loc=0, scale=0.01, size=synthetic_data[col].shape)
        df = pd.concat([df, synthetic_data], ignore_index=True)
        if len(df) > target_length:
            df = df.iloc[:target_length]
    return df


def generate_sliding_windows(data, sequence_length, step_size=1):
    """
    Generates sliding windows for sequence modeling.

    Args:
    -----
    data : np.ndarray
        Input data array where the last column is the target.
    sequence_length : int
        Number of timesteps in each window.
    step_size : int, optional
        Step size between consecutive windows (default is 1).

    Returns:
    --------
    tuple of np.ndarray
        Features and targets generated from sliding windows.
    """
    features, targets = [], []
    for i in range(0, len(data) - sequence_length, step_size):
        features.append(data[i:i + sequence_length, :-1])
        targets.append(data[i + sequence_length, -1])
    return np.array(features), np.array(targets)


def evaluate_genetic_model(data_path, learning_rate, batch_size, epochs, patience, fine_tune):
    """
    Evaluates and fine-tunes a genetic LSTM model.

    Args:
    -----
    data_path : str
        Path to the JSON data file.
    learning_rate : float
        Learning rate for the optimizer.
    batch_size : int
        Batch size for training.
    epochs : int
        Maximum number of epochs for training.
    patience : int
        Early stopping patience.
    fine_tune : int
        Number of layers to fine-tune.

    Raises:
    -------
    ValueError
        If data contains invalid or missing values.
    """
    try:
        sequence_length = 48
        model_file = '/app/models/enhanced_model.keras'
        model_file = os.path.abspath(model_file)

        feature_scaler_path = '/app/cache_json/feature_scaler.pkl'
        target_scaler_path = '/app/cache_json/target_scaler.pkl'

        preprocess_training_data(data_path)
        df = pd.read_json(data_path)

        if len(df) < 96:
            print("WARNING: Dataset has fewer than 96 rows. Adding synthetic rows...")
            df = add_synthetic_rows(df, target_length=96)

        raw_data = df[['value', 'value_rolling_mean_3', 'value_rolling_max_3', 'value_rolling_min_3',
                       'value_rolling_mean_6', 'value_rolling_max_6', 'value']].values

        features, targets = generate_sliding_windows(raw_data, sequence_length, step_size=sequence_length // 2)

        train_size = len(features) // 2
        train_features, train_targets = features[:train_size], targets[:train_size]
        eval_features, eval_targets = features[train_size:], targets[train_size:]

        if train_features.size == 0 or train_targets.size == 0 or eval_features.size == 0 or eval_targets.size == 0:
            raise ValueError("Features or targets have no values after processing.")

        train_features_flattened = train_features.reshape(-1, train_features.shape[-1])
        eval_features_flattened = eval_features.reshape(-1, eval_features.shape[-1])

        if os.path.exists(feature_scaler_path) and os.path.exists(target_scaler_path):
            feature_scaler = joblib.load(feature_scaler_path)
            target_scaler = joblib.load(target_scaler_path)
        else:
            feature_scaler = StandardScaler()
            target_scaler = StandardScaler()
            feature_scaler.fit(train_features_flattened)
            target_scaler.fit(train_targets.reshape(-1, 1))
            joblib.dump(feature_scaler, feature_scaler_path)
            joblib.dump(target_scaler, target_scaler_path)

        train_features_scaled = feature_scaler.transform(train_features_flattened).reshape(train_features.shape)
        train_targets_scaled = target_scaler.transform(train_targets.reshape(-1, 1)).flatten()

        eval_features_scaled = feature_scaler.transform(eval_features_flattened).reshape(eval_features.shape)
        eval_targets_scaled = target_scaler.transform(eval_targets.reshape(-1, 1)).flatten()

        custom_objects = {'mse': MeanSquaredError()}
        fog_model = tf.keras.models.load_model(model_file, custom_objects=custom_objects, compile=False)

        if fine_tune is not None:
            for layer in fog_model.layers[-int(fine_tune):]:
                layer.trainable = True

        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        fog_model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                                          restore_best_weights=True)

        fog_model.fit(train_features_scaled, train_targets_scaled, validation_data=(eval_features_scaled,
                                                                                    eval_targets_scaled),
                      epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=0)

        retrained_perf = fog_model.evaluate(eval_features_scaled, eval_targets_scaled, batch_size=batch_size, verbose=0)
        retrained_loss, retrained_mae = retrained_perf[:2] if isinstance(retrained_perf, list) else (retrained_perf,
                                                                                                     None)

        performances = {
            'retrained_model': {
                'loss': retrained_loss,
                'mae': retrained_mae,
            }
        }

        performances = {k: (None if pd.isna(v) else v) for k, v in performances.items()}
        print(json.dumps(performances))

    except Exception as e:
        error = {'error from evaluate_genetic_model': str(e)}
        print(json.dumps(error), file=sys.stderr)
        traceback.print_exc()


if __name__ == "__main__":
    """
    Entry point for the script. Accepts command-line arguments for training parameters.
    """
    data_path_par = sys.argv[1]
    learning_rate_par = float(sys.argv[2])
    batch_size_par = int(sys.argv[3])
    epochs_par = int(sys.argv[4])
    patience_par = int(sys.argv[5])
    fine_tune_par = int(sys.argv[6])
    evaluate_genetic_model(data_path_par, learning_rate_par, batch_size_par, epochs_par, patience_par,
                           fine_tune_par)
