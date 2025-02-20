import os
import pandas as pd
import tensorflow as tf
import numpy as np

from edge_node.data_selection import filter_data_by_interval_date, filter_data_by_day_date
from edge_node.edge_resources_paths import EdgeResourcesPaths
from edge_node.data_preprocessing import preprocess_data
from shared.logging_config import logger
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setting the environment variable to suppress TensorFlow low-level logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_initial_lstm_model(sequence_length=60, num_features=5):
    """
    Create and compile an enhanced LSTM model for time series forecasting.

    The model includes:
      - A 1D convolution layer to capture local temporal patterns.
      - Batch normalization and dropout for regularization.
      - Two LSTM layers with tanh activations.
      - Dense layers to learn non-linear relationships.

    Parameters:
        sequence_length (int): The length of the input sequences.
        num_features (int): The number of features per timestep.

    Returns:
        tf.keras.Model: A compiled LSTM model.
    """
    inputs = tf.keras.layers.Input(shape=(sequence_length, num_features), dtype=tf.float32)

    # Convolutional block to capture local patterns
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # LSTM layers for sequential modeling
    x = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)(x)
    x = tf.keras.layers.LSTM(128, activation='tanh')(x)

    # Dense layers for non-linear transformations
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='mse', metrics=["mae", "mse"])

    logger.info(f"Created model with input shape ({sequence_length}, {num_features})")
    return model


def create_sequences(data, sequence_length=60):
    """
    Convert a 2D array (n_samples, features) into a 3D array of sequences.
    Each sequence is a sliding window of fixed length.

    Parameters:
        data (np.array): 2D array of shape (n_samples, features).
        sequence_length (int): The length of each sequence.

    Returns:
        np.array: 3D array of shape (n_samples - sequence_length + 1, sequence_length, features)
    """
    num_sequences = len(data) - sequence_length + 1
    sequences = [data[i:i + sequence_length] for i in range(num_sequences)]
    sequences = np.array(sequences)
    logger.info(f"Created {sequences.shape[0]} sequences of length {sequence_length} with {data.shape[1]} features")
    return sequences


def pretrain_edge_model(edge_model_file_path: str, start_date: str, end_date: str, learning_rate: float,
                        batch_size: int, epochs: int, patience: int, fine_tune_layers: int, sequence_length=60):
    """
    Pretrain the model on data from a given period and evaluate on the same period.
    Save evaluation metrics before and after training.
    """
    # Load and filter the data for the specified period
    filter_data_by_interval_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime", start_date, end_date,
                                 EdgeResourcesPaths.FILTERED_DATA_PATH)
    preprocess_data(EdgeResourcesPaths.FILTERED_DATA_PATH, "datetime", "apparent power (kWh)")
    data = pd.read_csv(EdgeResourcesPaths.FILTERED_DATA_PATH)

    logger.info(f"Filtered data shape: {data.shape}")

    # Extract features and target
    X = data[[
        "value_rolling_mean_3", "value_rolling_max_3", "value_rolling_min_3",
        "value_rolling_mean_6", "value_rolling_max_6"
    ]].values
    y = data["value"].values

    logger.info(f"Features shape: {X.shape}, target length: {len(y)}")

    # Create sequences and corresponding targets
    num_sequences = len(X) - sequence_length + 1
    X_seq = create_sequences(X, sequence_length=sequence_length)
    y_seq = y[sequence_length - 1: sequence_length - 1 + num_sequences]
    logger.info(f"Created sequences: X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")

    # Split data into training and validation sets (keeping time order)
    split_index = int(len(X_seq) * 0.8)
    train_X, val_X = X_seq[:split_index], X_seq[split_index:]
    train_y, val_y = y_seq[:split_index], y_seq[split_index:]

    logger.info(f"Training set: {train_X.shape}, Validation set: {val_X.shape}")

    # Load the model
    custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
    model = tf.keras.models.load_model(edge_model_file_path, custom_objects=custom_objects)

    # Evaluate before training
    logger.info("Evaluating the model before training...")
    predictions_before = model.predict(val_X)
    evaluation_before = {
        "loss": float("inf"),
        "mae": float("inf"),
        "mse": float("inf"),
        "rmse": float("inf"),
        "r2": -float("inf")
    }
    try:
        evaluation_before.update({
            "loss": mean_squared_error(val_y, predictions_before),
            "mae": mean_absolute_error(val_y, predictions_before),
            "mse": mean_squared_error(val_y, predictions_before),
            "rmse": np.sqrt(mean_squared_error(val_y, predictions_before)),
            "r2": r2_score(val_y, predictions_before)
        })
    except Exception as e:
        logger.error(f"Evaluation before training failed, possibly due to untrained model: {e}")

    # Fine-tune the model (freeze all but the last fine_tune_layers)
    for layer in model.layers[:-fine_tune_layers]:
        layer.trainable = False

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=["mae", "mse"])

    # Train the model
    logger.info("Training the model...")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate after training
    logger.info("Evaluating the model after training...")
    predictions_after = model.predict(val_X)
    evaluation_after = {
        "loss": mean_squared_error(val_y, predictions_after),
        "mae": mean_absolute_error(val_y, predictions_after),
        "mse": mean_squared_error(val_y, predictions_after),
        "rmse": np.sqrt(mean_squared_error(val_y, predictions_after)),
        "r2": r2_score(val_y, predictions_after)
    }
    logger.info(f"Evaluation after training: {evaluation_after}")

    # Save metrics to a JSON file (or return them)
    metrics = {
        "before_training": evaluation_before,
        "after_training": evaluation_after
    }

    # Save the pretrained model
    pretrained_model_file_path = os.path.join(EdgeResourcesPaths.MODELS_FOLDER_PATH,
                                              EdgeResourcesPaths.RETRAINED_EDGE_MODEL_FILE_NAME)
    model.save(pretrained_model_file_path)
    logger.info(f"Pretrained model saved at: {pretrained_model_file_path}")

    return metrics


def retrain_edge_model(edge_model_file_path: str, date: str, learning_rate: float,
                       batch_size: int, epochs: int, patience: int, fine_tune_layers: int, sequence_length=60):
    """
    Retrain the model on the current day's data and evaluate on the next day's data.
    """
    # Define paths for current and next day data
    current_day_data_path = EdgeResourcesPaths.CURRENT_DAY_DATA_PATH
    next_day_data_path = EdgeResourcesPaths.NEXT_DAY_DATA_PATH

    # Process current day's data
    filter_data_by_day_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime", date, current_day_data_path)
    preprocess_data(current_day_data_path, "datetime", "apparent power (kWh)")

    next_day = pd.to_datetime(date) + pd.Timedelta(days=1)
    filter_data_by_day_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime",
                            next_day.strftime("%Y-%m-%d"), next_day_data_path)
    preprocess_data(next_day_data_path, "datetime", "apparent power (kWh)")

    train_data = pd.read_csv(current_day_data_path)
    eval_data = pd.read_csv(next_day_data_path)

    logger.info(f"Training data shape: {train_data.shape}, Evaluation data shape: {eval_data.shape}")

    # Prepare input and target data
    X_train = train_data[[
        "value_rolling_mean_3", "value_rolling_max_3", "value_rolling_min_3",
        "value_rolling_mean_6", "value_rolling_max_6"
    ]].values
    y_train = train_data["value"].values

    X_eval = eval_data[[
        "value_rolling_mean_3", "value_rolling_max_3", "value_rolling_min_3",
        "value_rolling_mean_6", "value_rolling_max_6"
    ]].values
    y_eval = eval_data["value"].values

    # Create sequences for both training and evaluation sets
    num_train_seq = len(X_train) - sequence_length + 1
    X_train_seq = create_sequences(X_train, sequence_length=sequence_length)
    y_train_seq = y_train[sequence_length - 1: sequence_length - 1 + num_train_seq]

    num_eval_seq = len(X_eval) - sequence_length + 1
    X_eval_seq = create_sequences(X_eval, sequence_length=sequence_length)
    y_eval_seq = y_eval[sequence_length - 1: sequence_length - 1 + num_eval_seq]

    logger.info(f"Training sequences: X_train_seq shape: {X_train_seq.shape}, y_train_seq shape: {y_train_seq.shape}")
    logger.info(f"Evaluation sequences: X_eval_seq shape: {X_eval_seq.shape}, y_eval_seq shape: {y_eval_seq.shape}")

    # Load the model
    custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
    model = tf.keras.models.load_model(edge_model_file_path, custom_objects=custom_objects)

    # Evaluate before retraining
    logger.info("Evaluating the model before retraining...")
    evaluation_before = model.evaluate(X_eval_seq, y_eval_seq, batch_size=batch_size, verbose=1)
    predictions_before = model.predict(X_eval_seq)
    r2_before = r2_score(y_eval_seq, predictions_before)
    rmse_before = np.sqrt(evaluation_before[1])
    metrics_before = {
        "loss": evaluation_before[0],
        "mae": evaluation_before[1],
        "mse": evaluation_before[2],
        "rmse": rmse_before,
        "r2": r2_before
    }

    logger.info(f"Metrics before retraining: {metrics_before}")

    # Fine-tune the model (freeze all but the last fine_tune_layers)
    for layer in model.layers[:-fine_tune_layers]:
        layer.trainable = False

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=["mae", "mse"])

    # Retrain the model
    logger.info("Retraining the model...")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    model.fit(
        X_train_seq, y_train_seq,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate after retraining
    logger.info("Evaluating the model after retraining...")
    evaluation_after = model.evaluate(X_eval_seq, y_eval_seq, batch_size=batch_size, verbose=1)
    predictions_after = model.predict(X_eval_seq)
    r2_after = r2_score(y_eval_seq, predictions_after)
    rmse_after = np.sqrt(evaluation_after[1])
    metrics_after = {
        "loss": evaluation_after[0],
        "mae": evaluation_after[1],
        "mse": evaluation_after[2],
        "rmse": rmse_after,
        "r2": r2_after
    }
    logger.info(f"Metrics after retraining: {metrics_after}")

    metrics = {
        "before_training": metrics_before,
        "after_training": metrics_after
    }

    # Save the retrained model
    retrained_edge_model_file_path = os.path.join(EdgeResourcesPaths.MODELS_FOLDER_PATH,
                                                  EdgeResourcesPaths.RETRAINED_EDGE_MODEL_FILE_NAME)
    model.save(retrained_edge_model_file_path)
    logger.info(f"Retrained model saved at: {retrained_edge_model_file_path}")

    return metrics
