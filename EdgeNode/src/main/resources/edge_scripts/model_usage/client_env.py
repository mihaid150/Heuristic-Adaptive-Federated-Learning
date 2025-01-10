"""
Script for training and retraining edge models using TensorFlow and custom layers.

The script preprocesses data, handles scalers, and trains an edge model with specified parameters.
It supports both initial training and retraining of models, evaluating the performance of the original
and retrained models.

Dependencies:
    - TensorFlow
    - scikit-learn
    - pandas
    - joblib
    - custom layers from `kan_usage`

Usage:
    python client_env.py <data_path> <model_file> <date> <mac> <learning_rate> <batch_size> <epochs> <patience>
    <fine_tune> <first_training|retrain>
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from keras.src.metrics.regression_metrics import MeanAbsoluteError
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
    Add synthetic rows to the DataFrame until it reaches the target length.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_length (int): Target number of rows.

    Returns:
        pd.DataFrame: Extended DataFrame with synthetic rows.
    """
    while len(df) < target_length:
        # Duplicate the existing DataFrame
        synthetic_data = df.sample(frac=1, replace=True, random_state=42).copy()

        # Add small random noise to the numeric columns
        for col in synthetic_data.select_dtypes(include=[np.number]).columns:
            synthetic_data[col] += np.random.normal(loc=0, scale=0.01, size=synthetic_data[col].shape)

        # Concatenate with the original DataFrame
        df = pd.concat([df, synthetic_data], ignore_index=True)

        # Truncate to the target length
        if len(df) > target_length:
            df = df.iloc[:target_length]

    return df


def prepare_features_and_target(df, sequence_length=48):
    """
        Prepares sliding windows for features and targets from the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with time series data.
            sequence_length (int): Length of each input sequence.

        Returns:
            tuple: (features, targets) arrays.
        """
    print(f"Initial DataFrame length: {len(df)}")

    # Ensure the dataset has at least 96 rows for current day and next day
    if len(df) < 96:
        print("WARNING: Dataset has fewer than 96 rows. Adding synthetic rows...")
        df = add_synthetic_rows(df, target_length=96)

    # Sliding window approach to generate features and targets
    def generate_sliding_windows(data, sequence_length, step_size=1):
        features, targets = [], []
        for i in range(0, len(data) - sequence_length, step_size):
            features.append(data[i:i + sequence_length, :-1])  # All but last column
            targets.append(data[i + sequence_length, -1])  # Last column as target
        return np.array(features), np.array(targets)

    # Prepare features and targets
    raw_data = df[['value', 'value_rolling_mean_3', 'value_rolling_max_3', 'value_rolling_min_3',
                   'value_rolling_mean_6', 'value_rolling_max_6', 'value']].values  # Add target column

    features, targets = generate_sliding_windows(raw_data, sequence_length)

    print(f"Generated features shape: {features.shape}")
    print(f"Generated target shape: {targets.shape}")

    if features.size == 0 or targets.size == 0:
        raise ValueError("Features or target have no values in prepare_features_and_target function")

    return features, targets


def manage_scalers(features, target, feature_scaler_path, target_scaler_path):
    """
        Load or create scalers and scale features and targets.

        Args:
            features (np.ndarray): Feature array.
            target (np.ndarray): Target array.
            feature_scaler_path (str): Path to save/load feature scaler.
            target_scaler_path (str): Path to save/load target scaler.

        Returns:
            tuple: Scaled features and target arrays.
        """

    # Check if scalers exist
    if os.path.exists(feature_scaler_path) and os.path.exists(target_scaler_path):
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        print("Scalers loaded from disk.")
    else:
        # Create and fit new scalers
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        num_samples, sequence_length, num_features = features.shape
        features_flattened = features.reshape(-1, num_features)

        feature_scaler.fit(features_flattened)
        target_scaler.fit(target.reshape(-1, 1))

        joblib.dump(feature_scaler, feature_scaler_path)
        joblib.dump(target_scaler, target_scaler_path)
        print("New scalers instantiated and saved to disk.")

    # Scale the features and target
    num_samples, sequence_length, num_features = features.shape
    features_flattened = features.reshape(-1, num_features)
    features_scaled = feature_scaler.transform(features_flattened).reshape(num_samples, sequence_length, num_features)
    target_scaled = target_scaler.transform(target.reshape(-1, 1)).flatten()

    return features_scaled, target_scaled


def edge_model_preparation(features, target, model_file, fine_tune, learning_rate, patience, epochs, batch_size):
    """
        Prepares and fine-tunes the edge model using the provided features and target.

        Args:
            features (np.ndarray): Input features array of shape (samples, timesteps, features).
            target (np.ndarray): Target values array of shape (samples,).
            model_file (str): Path to the saved model file.
            fine_tune (int): Number of layers to fine-tune from the end of the model.
            learning_rate (float): Learning rate for the Adam optimizer.
            patience (int): Number of epochs to wait before early stopping.
            epochs (int): Maximum number of epochs to train.
            batch_size (int): Batch size for training.

        Returns:
            tuple: Trained model, test features, test targets, and custom objects dictionary.
        """

    # Scaler paths
    feature_scaler_path = '/app/cache_json/feature_scaler.pkl'
    target_scaler_path = '/app/cache_json/target_scaler.pkl'

    # Manage scalers
    features_scaled, target_scaled = manage_scalers(features, target, feature_scaler_path, target_scaler_path)

    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42)

    # Load the model with custom layers (DenseKAN and SymbolicKAN)
    custom_objects = {'mse': MeanSquaredError()}
    edge_model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)

    # Fine-tune the model if specified
    if fine_tune > 0:
        for layer in edge_model.layers[-int(fine_tune):]:
            layer.trainable = True

    # Compile the model with Adam optimizer, MSE loss, and MAE metric
    optimizer = Adam(learning_rate=learning_rate)
    edge_model.compile(optimizer=optimizer, loss='mse', metrics=[MeanAbsoluteError()])

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Train the model
    edge_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size,
                   callbacks=[early_stopping, reduce_lr], verbose=0)

    return edge_model, x_test, y_test, custom_objects


def retrain_edge_model(new_data_path, model_file, date, mac, learning_rate, batch_size, epochs, patience, fine_tune):
    """
        Retrains an edge model with new data.

        Args:
            new_data_path (str): Path to the new training data.
            model_file (str): Path to the existing model to be retrained.
            date (str): Date of the retraining session.
            mac (str): Unique identifier for the device.
            learning_rate (float): Learning rate for training.
            batch_size (int): Batch size for training.
            epochs (int): Maximum number of epochs for training.
            patience (int): Patience parameter for early stopping.
            fine_tune (int): Number of layers to fine-tune from the end of the model.

        Outputs:
            Prints JSON of model performance metrics and saves the retrained model.
        """

    try:
        # Preprocess the data
        preprocess_training_data(new_data_path)
        df = pd.read_json(new_data_path)

        features, target = prepare_features_and_target(df)

        if features.size == 0 or target.size == 0 or len(features) != len(target):
            raise ValueError("Features or target have no values in retrain_edge_model function")

        # Add noise to features for data augmentation
        features += 0.01 * np.random.normal(size=features.shape)

        edge_model, x_test, y_test, custom_objects = edge_model_preparation(features, target, model_file, fine_tune,
                                                                            learning_rate, patience, epochs, batch_size)
        # Evaluate retrained performance using additional metrics
        edge_performance = edge_model.evaluate(x_test, y_test, verbose=0)
        edge_loss, edge_mae = edge_performance
        edge_y_pred = edge_model.predict(x_test).squeeze()

        # Compute additional metrics like R² and RMSE
        edge_r2 = r2_score(y_test, edge_y_pred)
        edge_rmse = np.sqrt(mean_squared_error(y_test, edge_y_pred))

        print(f"Retrained model performance:")
        print(f"Loss (MSE): {edge_loss}")
        print(f"Mean Absolute Error (MAE): {edge_mae}")
        print(f"R-squared (R²): {edge_r2}")
        print(f"Root Mean Squared Error (RMSE): {edge_rmse}")

        # Compare with the original model
        original_model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
        original_performance = original_model.evaluate(x_test, y_test, verbose=0)
        original_loss, original_mae = original_performance
        original_y_pred = original_model.predict(x_test)
        original_y_pred = original_y_pred.squeeze()
        original_r2 = r2_score(y_test, original_y_pred)
        original_rmse = np.sqrt(mean_squared_error(y_test, original_y_pred))

        print(f"Original model performance:")
        print(f"Loss (MSE): {original_loss}")
        print(f"Mean Absolute Error (MAE): {original_mae}")
        print(f"R-squared (R²): {original_r2}")
        print(f"Root Mean Squared Error (RMSE): {original_rmse}")

        # Store performances for comparison
        performances = {
            'original_model': {
                'loss': original_loss,
                'mae': original_mae,
                'r2': original_r2,
                'rmse': original_rmse
            },
            'retrained_model': {
                'loss': edge_loss,
                'mae': edge_mae,
                'r2': edge_r2,
                'rmse': edge_rmse
            }
        }

        # Replace NaN with None if necessary
        performances = {k: (None if pd.isna(v) else v) for k, v in performances.items()}

        # Save retrained model
        model_save_path = "/app/models/"
        retrained_model_path = os.path.join(model_save_path, f'local_model_{mac}_{date.replace("-", "_")}.keras')
        edge_model.save(retrained_model_path)

        # Output performances as JSON
        print(json.dumps(performances))

    except Exception as e:
        error = {'error from retrain_edge_model': str(e)}
        print(json.dumps(error), file=sys.stderr)
        traceback.print_exc()


def first_training(new_data_path, model_file, date, mac, learning_rate, batch_size, epochs, patience, fine_tune):
    """
       Trains a new edge model for the first time using the provided data.

       Args:
           new_data_path (str): Path to the training data.
           model_file (str): Path to the model file to initialize training.
           date (str): Date of the training session.
           mac (str): Unique identifier for the device.
           learning_rate (float): Learning rate for training.
           batch_size (int): Batch size for training.
           epochs (int): Maximum number of epochs for training.
           patience (int): Patience parameter for early stopping.
           fine_tune (int): Number of layers to fine-tune from the end of the model.

       Outputs:
           Prints JSON of model performance metrics and saves the trained model.
       """

    try:
        # Preprocess the data
        preprocess_training_data(new_data_path)
        df = pd.read_json(new_data_path)

        features, target = prepare_features_and_target(df)

        if features.size == 0 or target.size == 0 or len(features) != len(target):
            raise ValueError("Features or target have no values in first_training function")

        edge_model, x_test, y_test, custom_objects = edge_model_preparation(features, target, model_file, fine_tune,
                                                                            learning_rate, patience, epochs, batch_size)

        # Save the trained model
        model_save_path = "/app/models/"
        trained_model_path = os.path.join(model_save_path, f'local_model_{mac}_{date.replace("-", "_")}.keras')
        edge_model.save(trained_model_path)

        # Evaluate retrained performance using additional metrics
        edge_performance = edge_model.evaluate(x_test, y_test, verbose=0)
        edge_loss, edge_mae = edge_performance
        edge_y_pred = edge_model.predict(x_test).squeeze()

        # Compute additional metrics like R² and RMSE
        edge_r2 = r2_score(y_test, edge_y_pred)
        edge_rmse = np.sqrt(mean_squared_error(y_test, edge_y_pred))

        print(f"Retrained model performance:")
        print(f"Loss (MSE): {edge_loss}")
        print(f"Mean Absolute Error (MAE): {edge_mae}")
        print(f"R-squared (R²): {edge_r2}")
        print(f"Root Mean Squared Error (RMSE): {edge_rmse}")

        # Store performances for comparison
        performances = {
            'original_model': {
                'loss': 999999,
                'mae': 999999,
                'r2': 999999,
                'rmse': 999999
            },
            'retrained_model': {
                'loss': edge_loss,
                'mae': edge_mae,
                'r2': edge_r2,
                'rmse': edge_rmse
            }
        }

        # Replace NaN with None if necessary
        performances = {k: (None if pd.isna(v) else v) for k, v in performances.items()}

        # Output performances as JSON
        print(json.dumps(performances))

    except Exception as e:
        error = {'error from first_training': str(e)}
        print(json.dumps(error), file=sys.stderr)
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 10:
        print(json.dumps({'error main client_env': 'Not all arguments are provided. Expected 10 arguments'}))
    else:
        data_path_param = sys.argv[1]
        model_file_param = sys.argv[2]
        date_param = sys.argv[3]
        mac_param = sys.argv[4]
        learning_rate_param = float(sys.argv[5])
        batch_size_param = int(sys.argv[6])
        epochs_param = int(sys.argv[7])
        patience_param = int(sys.argv[8])
        fine_tune_param = int(sys.argv[9])
        first = sys.argv[10]

        if first == "first_training":
            first_training(data_path_param, model_file_param, date_param, mac_param, learning_rate_param,
                           batch_size_param, epochs_param, patience_param, fine_tune_param)
        else:
            retrain_edge_model(data_path_param, model_file_param, date_param, mac_param, learning_rate_param,
                               batch_size_param, epochs_param, patience_param, fine_tune_param)
