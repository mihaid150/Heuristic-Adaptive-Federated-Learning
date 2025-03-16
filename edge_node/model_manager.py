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


def compute_metrics(y_true, y_pred):
    """
    Compute additional evaluation metrics.
    Return a dictionary with:
        - mse: Mean Squared Error
        - mae: Mean Absolute Error
        - r2: R-squared score
        - logcosh: Log-Cosh loss (averaged over samples)
        - huber: Huber Loss (averaged over samples)
        - msle: Mean Squared Log Error
    """
    mse_val = float(mean_squared_error(y_true, y_pred))
    mae_val = float(mean_absolute_error(y_true, y_pred))
    r2_val = float(r2_score(y_true, y_pred))
    logcosh_val = float(np.mean(np.log(np.cosh(y_pred - y_true))))

    huber_loss_fn = tf.keras.losses.Huber()
    huber_val = float(huber_loss_fn(y_true, y_pred).numpy())

    # msle: ensure no negative values by using log1p
    msle_val = float(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

    return {
        "mse": mse_val,
        "mae": mae_val,
        "r2": r2_val,
        "logcosh": logcosh_val,
        "huber": huber_val,
        "msle": msle_val
    }


def create_feature_sequences_with_padding(data, sequence_length):
    """
    Convert a 2D array (n_samples, features) into a 3D array of sequences,
    where each sequence has a fixed length equal to `sequence_length` timesteps.

    A sliding window of length `sequence_length` is used if possible.
    If the entire data is shorter than the desired window length, the data
    is padded with zeros to reach `sequence_length`. Additionally, if only one
    sequence is generated, it is duplicated to ensure at least 2 sequences.

    Parameters:
        data (np.array): 2D array of shape (n_samples, num_features).
        sequence_length (int): The desired fixed length for the sequences.

    Returns:
        np.array: 3D array of shape (num_sequences, sequence_length, num_features).
    """
    n_samples, num_features = data.shape

    if n_samples >= sequence_length:
        num_windows = n_samples - sequence_length + 1
        sequences = [data[i:i + sequence_length] for i in range(num_windows)]
        sequences = np.array(sequences)
        logger.info(
            f"Created {sequences.shape[0]} sequences of fixed length {sequence_length} with {num_features} features.")
    else:
        padding_needed = sequence_length - n_samples
        pad = np.zeros((padding_needed, num_features))
        padded_data = np.vstack([data, pad])
        sequences = np.expand_dims(padded_data, axis=0)
        logger.warning(f"Data length ({n_samples}) is less than the desired window length ({sequence_length}). "
                       f"Returning a single padded sequence of fixed length {sequence_length}.")

    # Ensure at least two sequences are available for metric computation.
    if sequences.shape[0] < 2:
        sequences = np.concatenate([sequences, sequences], axis=0)
        logger.warning(
            "Only one sequence was created; replicating it to ensure at least two sequences for metrics computation.")

    return sequences


def create_target_sequences(targets: np.array, sequence_length: int, min_sequences: int = 2) -> np.array:
    """
    Convert a 1D target array into a sequence array that matches the number of sequences
    produced for the features by a padding-based sequence function.

    When len(targets) >= sequence_length, a sliding window is used to generate target values.
    Otherwise, if there are not enough samples:
        - If at least 2 samples exist, perform linear interpolation to generate min_sequences values.
        - If only one sample exists, replicate it min_sequences times.

    Parameters:
        targets (np.array): 1D array of target values.
        sequence_length (int): Desired sequence length (used to decide sliding window vs. interpolation).
        min_sequences (int): Minimum number of sequences to output when interpolation is used (default: 2).

    Returns:
        np.array: 1D array of target values for each sequence.
    """
    if len(targets) >= sequence_length:
        # Use sliding window approach
        n_seq = len(targets) - sequence_length + 1
        return targets[sequence_length - 1: sequence_length - 1 + n_seq]
    else:
        # Not enough samples: use interpolation if possible
        if len(targets) >= 2:
            # Create evenly spaced indices between 0 and len(targets)-1 for min_sequences points.
            interp_indices = np.linspace(0, len(targets) - 1, num=min_sequences)
            # Use np.interp to generate interpolated target values.
            interpolated_targets = np.interp(interp_indices, np.arange(len(targets)), targets)
            return interpolated_targets
        elif len(targets) == 1:
            # Only one target value: replicate it.
            return np.array([targets[0]] * min_sequences)
        else:
            # No targets available; return an empty array.
            return np.array([])


def pretrain_edge_model(edge_model_file_path: str, start_date: str, end_date: str, learning_rate: float,
                        batch_size: int, epochs: int, patience: int, fine_tune_layers: int, sequence_length):
    # Load and filter data
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

    # Create sequences for features and targets using dedicated functions
    X_seq = create_feature_sequences_with_padding(X, sequence_length=sequence_length)
    y_seq = create_target_sequences(y, sequence_length=sequence_length)
    logger.info(f"Created sequences: X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")

    split_index = int(len(X_seq) * 0.8)
    train_X, val_X = X_seq[:split_index], X_seq[split_index:]
    train_y, val_y = y_seq[:split_index], y_seq[split_index:]
    logger.info(f"Training set: {train_X.shape}, Validation set: {val_X.shape}")

    custom_objects = {
        "LogCosh": tf.keras.losses.LogCosh(),
        "mse": tf.keras.losses.MeanSquaredError(),
        "Huber": tf.keras.losses.Huber()
    }
    model = tf.keras.models.load_model(edge_model_file_path, custom_objects=custom_objects)

    logger.info("Evaluating the model before training...")
    predictions_before = model.predict(val_X)
    eval_before = compute_metrics(val_y, predictions_before)

    # Fine-tune: freeze all but the last fine_tune_layers
    for layer in model.layers[:-fine_tune_layers]:
        layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.LogCosh())

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

    logger.info("Evaluating the model after training...")
    predictions_after = model.predict(val_X)
    eval_after = compute_metrics(val_y, predictions_after)
    logger.info(f"Evaluation after training: {eval_after}")

    metrics = {
        "before_training": eval_before,
        "after_training": eval_after
    }

    pretrained_model_file_path = os.path.join(EdgeResourcesPaths.MODELS_FOLDER_PATH,
                                              EdgeResourcesPaths.RETRAINED_EDGE_MODEL_FILE_NAME)
    model.save(pretrained_model_file_path)
    logger.info(f"Pretrained model saved at: {pretrained_model_file_path}")
    return metrics


def retrain_edge_model(edge_model_file_path: str, date: str, learning_rate: float,
                       batch_size: int, epochs: int, patience: int, fine_tune_layers: int, sequence_length):
    # Process current and next day data
    current_day_data_path = EdgeResourcesPaths.CURRENT_DAY_DATA_PATH
    next_day_data_path = EdgeResourcesPaths.NEXT_DAY_DATA_PATH
    filter_data_by_day_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime", date, current_day_data_path)
    preprocess_data(current_day_data_path, "datetime", "apparent power (kWh)")
    next_day = pd.to_datetime(date) + pd.Timedelta(days=1)
    filter_data_by_day_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime",
                            next_day.strftime("%Y-%m-%d"), next_day_data_path)
    preprocess_data(next_day_data_path, "datetime", "apparent power (kWh)")
    train_data = pd.read_csv(current_day_data_path)
    eval_data = pd.read_csv(next_day_data_path)
    logger.info(f"Training data shape: {train_data.shape}, Evaluation data shape: {eval_data.shape}")

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

    # Create sequences for training and evaluation data
    X_train_seq = create_feature_sequences_with_padding(X_train, sequence_length=sequence_length)
    y_train_seq = create_target_sequences(y_train, sequence_length=sequence_length)
    X_eval_seq = create_feature_sequences_with_padding(X_eval, sequence_length=sequence_length)
    y_eval_seq = create_target_sequences(y_eval, sequence_length=sequence_length)

    logger.info(f"Training sequences: X_train_seq shape: {X_train_seq.shape}, y_train_seq shape: {y_train_seq.shape}")
    logger.info(f"Evaluation sequences: X_eval_seq shape: {X_eval_seq.shape}, y_eval_seq shape: {y_eval_seq.shape}")

    custom_objects = {
        "LogCosh": tf.keras.losses.LogCosh(),
        "mse": tf.keras.losses.MeanSquaredError(),
        "Huber": tf.keras.losses.Huber()
    }
    model = tf.keras.models.load_model(edge_model_file_path, custom_objects=custom_objects)

    logger.info("Evaluating the model before retraining...")
    evaluation_before = compute_metrics(y_eval_seq, model.predict(X_eval_seq))
    logger.info(f"Metrics before retraining: {evaluation_before}")

    for layer in model.layers[:-fine_tune_layers]:
        layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.LogCosh())
    logger.info("Retraining the model...")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    model.fit(
        X_train_seq, y_train_seq,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )

    logger.info("Evaluating the model after retraining...")
    evaluation_after = compute_metrics(y_eval_seq, model.predict(X_eval_seq))
    logger.info(f"Metrics after retraining: {evaluation_after}")

    metrics = {
        "before_training": evaluation_before,
        "after_training": evaluation_after
    }

    retrained_edge_model_file_path = os.path.join(EdgeResourcesPaths.MODELS_FOLDER_PATH,
                                                  EdgeResourcesPaths.RETRAINED_EDGE_MODEL_FILE_NAME)
    model.save(retrained_edge_model_file_path)
    logger.info(f"Retrained model saved at: {retrained_edge_model_file_path}")

    return metrics


def evaluate_edge_model(edge_model_file_path: str, date: str, sequence_length):
    current_day_data_path = EdgeResourcesPaths.CURRENT_DAY_DATA_PATH

    filter_data_by_day_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime", date, current_day_data_path)
    preprocess_data(current_day_data_path, "datetime", "apparent power (kWh)")

    eval_data = pd.read_csv(current_day_data_path)
    logger.info(f"Evaluation data shape: {eval_data.shape}")

    X_eval = eval_data[["value_rolling_mean_3", "value_rolling_max_3", "value_rolling_min_3",
                        "value_rolling_mean_6", "value_rolling_max_6"]].values
    y_eval = eval_data["value"].values

    X_eval_seq = create_feature_sequences_with_padding(X_eval, sequence_length=sequence_length)
    y_eval_seq = create_target_sequences(y_eval, sequence_length=sequence_length)

    logger.info(f"Evaluation sequences: X_eval_seq shape: {X_eval_seq.shape}, y_eval_seq shape: {y_eval_seq.shape}")

    custom_objects = {
        "LogCosh": tf.keras.losses.LogCosh(),
        "mse": tf.keras.losses.MeanSquaredError(),
        "Huber": tf.keras.losses.Huber()
    }
    model = tf.keras.models.load_model(edge_model_file_path, custom_objects=custom_objects)

    logger.info("Evaluating the model on next day's data...")
    predictions = model.predict(X_eval_seq)
    metrics = compute_metrics(y_eval_seq, predictions)
    logger.info(f"Evaluation metrics: {metrics}")
    predictions_flat = predictions.flatten()
    real_values = y_eval_seq.flatten()
    prediction_pairs = list(zip(real_values.tolist(), predictions_flat.tolist()))
    return metrics, prediction_pairs


def validate_data_size(start_date: str, end_date: str, sequence_length: int) -> bool:
    if start_date is not None:
        logger.info(f"Validating data size for interval filtering: start_date={start_date}, end_date={end_date}")
        filter_data_by_interval_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime", start_date, end_date,
                                     EdgeResourcesPaths.FILTERED_DATA_PATH)
        preprocess_data(EdgeResourcesPaths.FILTERED_DATA_PATH, "datetime", "apparent power (kWh)")
        data = pd.read_csv(EdgeResourcesPaths.FILTERED_DATA_PATH)
        logger.info(f"Filtered data shape: {data.shape}")

        if sequence_length >= data.shape[0]:
            logger.error(f"Sequence length ({sequence_length}) is greater than or equal to the number of rows ({data.shape[0]}).")
            return False
        else:
            logger.info(f"Data size is sufficient for sequence length {sequence_length}.")
            return True
    else:
        logger.info(f"Validating data size for day-based filtering for date: {end_date}")
        current_day_data_path = EdgeResourcesPaths.CURRENT_DAY_DATA_PATH
        next_day_data_path = EdgeResourcesPaths.NEXT_DAY_DATA_PATH

        filter_data_by_day_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime", end_date, current_day_data_path)
        preprocess_data(current_day_data_path, "datetime", "apparent power (kWh)")
        next_day = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        filter_data_by_day_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime",
                                next_day.strftime("%Y-%m-%d"), next_day_data_path)
        preprocess_data(next_day_data_path, "datetime", "apparent power (kWh)")
        train_data = pd.read_csv(current_day_data_path)
        eval_data = pd.read_csv(next_day_data_path)
        logger.info(f"Training data shape: {train_data.shape}, Evaluation data shape: {eval_data.shape}")

        if sequence_length >= train_data.shape[0] or sequence_length >= eval_data.shape[0]:
            logger.error(f"Sequence length ({sequence_length}) is greater than or equal to the training or evaluation data size.")
            return False
        else:
            logger.info(f"Data sizes are sufficient for sequence length {sequence_length} (both training and evaluation).")
            return True


