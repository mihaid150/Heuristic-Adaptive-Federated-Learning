# edge_node/model_manager.py

import os
import pandas as pd
import tensorflow as tf
import numpy as np

from edge_node.data_selection import filter_data_by_interval_date
from edge_node.edge_resources_paths import EdgeResourcesPaths
from edge_node.data_preprocessing import preprocess_data
from shared.utils import required_columns
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
    is padded with -1 (mask value) to reach `sequence_length`. Additionally, if only one
    sequence is generated, it is duplicated to ensure at least 2 sequences.
    """
    n_samples, num_features = data.shape
    mask_value = -1  # using -1 as the mask value

    if n_samples >= sequence_length:
        num_windows = n_samples - sequence_length + 1
        sequences = [data[i:i + sequence_length] for i in range(num_windows)]
        sequences = np.array(sequences)
        logger.info(
            f"Created {sequences.shape[0]} sequences of fixed length {sequence_length} with {num_features} features.")
    else:
        padding_needed = sequence_length - n_samples
        pad = np.full((padding_needed, num_features), mask_value)  # pad with -1 instead of 0
        padded_data = np.vstack([data, pad])
        sequences = np.expand_dims(padded_data, axis=0)
        logger.warning(f"Data length ({n_samples}) is less than the desired window length ({sequence_length}). "
                       f"Returning a single padded sequence of fixed length {sequence_length}.")

    if sequences.shape[0] < 2:
        sequences = np.concatenate([sequences, sequences], axis=0)
        logger.warning("Only one sequence was created; replicating it to ensure at least two sequences for metrics "
                       "computation.")

    return sequences


def create_target_sequences(targets: np.array, sequence_length: int, min_sequences: int = 2, mask_value=-1) -> np.array:
    """
    Convert a 1D target array into a sequence of target values corresponding to each sequence produced by the
    feature padding function. When len(targets) >= sequence_length, a sliding window is used and the target
    is taken as the last value in each window. Otherwise, if there are not enough samples,
        - The target array is padded with mask_value (-1) to reach sequence_length,
        - And the final (padded) target value is replicated min_sequences times.

    Parameters:
        targets (np.array): 1D array of target values.
        sequence_length (int): Desired sequence length.
        min_sequences (int): Minimum number of sequences to output when interpolation/padding is used.
        mask_value (int/float): The value used for padding (here, -1).

    Returns:
        np.array: 1D array of target values (one per sequence).
    """
    if len(targets) >= sequence_length:
        n_seq = len(targets) - sequence_length + 1
        return targets[sequence_length - 1: sequence_length - 1 + n_seq]
    else:
        padding_needed = sequence_length - len(targets)
        padded_targets = np.concatenate([targets, np.full((padding_needed,), mask_value)])
        # Use the last value of the padded sequence as the target.
        return np.array([padded_targets[-1]] * min_sequences)


def post_preprocessing_padding(data_file_path: str, required_length: int, mask_value: float = -1):
    df = pd.read_csv(data_file_path)
    current_rows = len(df)
    if required_length > current_rows > 0:
        last_row = df.iloc[-1].copy()
        last_row["synthetic"] = True  # mark as synthetic if needed
        missing = 2 * required_length - current_rows
        synthetic_rows = [last_row.copy() for _ in range(missing)]
        df_synthetic = pd.DataFrame(synthetic_rows)
        df = pd.concat([df, df_synthetic], ignore_index=True)
        df.to_csv(data_file_path, index=False)
    return data_file_path


def determine_sequence_length(dataframe, target_length=144):
    """
    Determine the sequence length based on available data.
    If available rows >= target_length, use target_length.
    Otherwise, use the available number of rows.
    """
    available = dataframe.shape[0]
    chosen_length = target_length if available >= target_length else available
    logger.info(f"Determined sequence length: {chosen_length} (available rows: {available}, target: {target_length})")
    return chosen_length


def pretrain_edge_model(edge_model_file_path: str, start_date: str, end_date: str, learning_rate: float,
                        batch_size: int, epochs: int, patience: int, fine_tune_layers: int, sequence_length):
    # Load and filter data
    filter_data_by_interval_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime", start_date, end_date,
                                 EdgeResourcesPaths.FILTERED_DATA_PATH, print_loggings=False)
    preprocess_data(EdgeResourcesPaths.FILTERED_DATA_PATH, "datetime", "apparent power (kWh)")
    data = pd.read_csv(EdgeResourcesPaths.FILTERED_DATA_PATH)
    logger.info(f"Filtered data shape: {data.shape}")

    feature_columns = required_columns.copy()
    feature_columns.remove("value")

    # Extract features and target
    X = data[feature_columns].values
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
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())

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


def retrain_edge_model(edge_model_file_path: str, start_date: str, learning_rate: float,
                       batch_size: int, epochs: int, patience: int, fine_tune_layers: int, target_sequence_length=144):
    """
    Retrain the edge model using two days for training and two days for evaluation.
    'start_date' is the first training day.
    Training period covers start_date and the next day.
    Evaluation period covers the two days following the training period.
    """
    # Define dates for training and evaluation
    training_day1 = start_date
    training_day2 = (pd.to_datetime(start_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    evaluation_day1 = (pd.to_datetime(start_date) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    evaluation_day2 = (pd.to_datetime(start_date) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    # Paths for merged training data and evaluation data
    training_data_path = EdgeResourcesPaths.RETRAINING_DAYS_DATA_PATH  # new path for merged training data
    evaluation_data_path = EdgeResourcesPaths.EVALUATION_DAYS_DATA_PATH  # new path for merged evaluation data

    # For training, filter data from training_day1 to training_day2 (inclusive)
    filter_data_by_interval_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime",
                                 training_day1, training_day2, training_data_path)
    preprocess_data(training_data_path, "datetime", "apparent power (kWh)")
    post_preprocessing_padding(training_data_path, target_sequence_length)
    train_data = pd.read_csv(training_data_path)
    logger.info(f"Training data shape (2 days): {train_data.shape}")

    # For evaluation, filter data from evaluation_day1 to evaluation_day2 (inclusive)
    filter_data_by_interval_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime",
                                 evaluation_day1, evaluation_day2, evaluation_data_path)
    preprocess_data(evaluation_data_path, "datetime", "apparent power (kWh)")
    post_preprocessing_padding(evaluation_data_path, target_sequence_length)
    eval_data = pd.read_csv(evaluation_data_path)
    logger.info(f"Evaluation data shape (2 days): {eval_data.shape}")

    feature_columns = required_columns.copy()
    feature_columns.remove("value")

    # Extract features and target for training and evaluation
    X_train = train_data[feature_columns].values
    y_train = train_data["value"].values
    X_eval = eval_data[feature_columns].values
    y_eval = eval_data["value"].values

    # Dynamically determine sequence length for training and evaluation data
    train_sequence_length = determine_sequence_length(train_data, target_length=target_sequence_length)
    eval_sequence_length = determine_sequence_length(eval_data, target_length=target_sequence_length)
    logger.info(f"Using training sequence length: {train_sequence_length} and evaluation sequence length: "
                f"{eval_sequence_length}")

    # Create sequences using the determined sequence lengths.
    X_train_seq = create_feature_sequences_with_padding(X_train, sequence_length=train_sequence_length)
    y_train_seq = create_target_sequences(y_train, sequence_length=train_sequence_length)
    X_eval_seq = create_feature_sequences_with_padding(X_eval, sequence_length=eval_sequence_length)
    y_eval_seq = create_target_sequences(y_eval, sequence_length=eval_sequence_length)

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
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())
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


def evaluate_edge_model(edge_model_file_path: str, start_date: str, sequence_length):
    """
    Evaluate the edge model using seven days of data.
    'start_date' is treated as the first evaluation day, and we collect data for seven consecutive days.
    """
    # Define evaluation period: from start_date to start_date + 6 days.
    evaluation_day1 = start_date
    evaluation_day7 = (pd.to_datetime(evaluation_day1) + pd.Timedelta(days=6)).strftime("%Y-%m-%d")

    evaluation_data_path = EdgeResourcesPaths.EVALUATION_DAYS_DATA_PATH
    # Filter data for the seven-day period.
    filter_data_by_interval_date(
        EdgeResourcesPaths.INPUT_DATA_PATH, "datetime",
        evaluation_day1, evaluation_day7,
        evaluation_data_path
    )
    preprocess_data(evaluation_data_path, "datetime", "apparent power (kWh)")
    post_preprocessing_padding(evaluation_data_path, sequence_length)
    eval_data = pd.read_csv(evaluation_data_path)
    logger.info(f"Evaluation data shape (7 days): {eval_data.shape}")

    feature_columns = required_columns.copy()
    feature_columns.remove("value")

    X_eval = eval_data[feature_columns].values
    y_eval = eval_data["value"].values

    # Use the same window length (144) as during training.
    X_eval_seq = create_feature_sequences_with_padding(X_eval, sequence_length=sequence_length)
    y_eval_seq = create_target_sequences(y_eval, sequence_length=sequence_length)
    logger.info(f"Evaluation sequences: X_eval_seq shape: {X_eval_seq.shape}, y_eval_seq shape: {y_eval_seq.shape}")

    custom_objects = {
        "LogCosh": tf.keras.losses.LogCosh(),
        "mse": tf.keras.losses.MeanSquaredError(),
        "Huber": tf.keras.losses.Huber()
    }
    model = tf.keras.models.load_model(edge_model_file_path, custom_objects=custom_objects)

    logger.info("Evaluating the model on seven days of data...")
    predictions = model.predict(X_eval_seq)
    metrics = compute_metrics(y_eval_seq, predictions)
    logger.info(f"Evaluation metrics: {metrics}")
    predictions_flat = predictions.flatten()
    real_values = y_eval_seq.flatten()
    prediction_pairs = list(zip(real_values.tolist(), predictions_flat.tolist()))
    return metrics, prediction_pairs


def validate_data_size(start_date: str, end_date: str, sequence_length: int, is_training_validation: bool = True) -> (
        bool):
    if start_date is not None:
        logger.info(f"Validating data size for interval filtering: start_date={start_date}, end_date={end_date}")
        filter_data_by_interval_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime", start_date, end_date,
                                     EdgeResourcesPaths.FILTERED_DATA_PATH, print_loggings=False)
        preprocess_data(EdgeResourcesPaths.FILTERED_DATA_PATH, "datetime", "apparent power (kWh)")
        data = pd.read_csv(EdgeResourcesPaths.FILTERED_DATA_PATH)
        logger.info(f"Filtered data shape: {data.shape}")

        if sequence_length >= data.shape[0]:
            logger.error(f"Sequence length ({sequence_length}) is greater than or equal to the number of rows "
                         f"({data.shape[0]}).")
            return False
        else:
            logger.info(f"Data size is sufficient for sequence length {sequence_length}.")
            return True
    else:
        if is_training_validation:
            training_days_data_path = EdgeResourcesPaths.RETRAINING_DAYS_DATA_PATH
            evaluation_day_data_path = EdgeResourcesPaths.EVALUATION_DAYS_DATA_PATH

            training_day1 = end_date
            training_day2 = (pd.to_datetime(end_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            evaluation_day1 = (pd.to_datetime(end_date) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            evaluation_day2 = (pd.to_datetime(end_date) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            logger.info(f"Training validation data size of 2-day-based filtering for dates training day 1 "
                        f"{training_day1} and training day 2 {training_day2} with evaluation day 1 {evaluation_day1} "
                        f"and evaluation day 2 {evaluation_day2}")

            filter_data_by_interval_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime", training_day1,
                                         training_day2, training_days_data_path)
            preprocess_data(training_days_data_path, "datetime", "apparent power (kWh)")

            filter_data_by_interval_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime",
                                         evaluation_day1, evaluation_day2,
                                         evaluation_day_data_path)

            preprocess_data(evaluation_day_data_path, "datetime", "apparent power (kWh)")

            train_data = pd.read_csv(training_days_data_path)
            eval_data = pd.read_csv(evaluation_day_data_path)
            logger.info(f"Training data shape: {train_data.shape}, Evaluation data shape: {eval_data.shape}")

            if sequence_length > train_data.shape[0] or sequence_length / 2 >= eval_data.shape[0]:
                logger.error(f"Sequence length ({sequence_length}) is greater than or equal to the training or "
                             f"evaluation data size.")
                return False
            else:
                logger.info(f"Data sizes are sufficient for sequence length {sequence_length} (both training and "
                            f"evaluation).")
                return True
        else:
            logger.info(f"Evaluation validation data size for day-based filtering for date: {end_date}")

            evaluation_day1 = end_date
            evaluation_day7 = (pd.to_datetime(evaluation_day1) + pd.Timedelta(days=6)).strftime("%Y-%m-%d")

            evaluation_data_path = EdgeResourcesPaths.EVALUATION_DAYS_DATA_PATH
            # Filter data for the seven-day period.
            filter_data_by_interval_date(
                EdgeResourcesPaths.INPUT_DATA_PATH, "datetime",
                evaluation_day1, evaluation_day7,
                evaluation_data_path
            )

            preprocess_data(evaluation_data_path, "datetime", "apparent power (kWh)")

            train_data = pd.read_csv(evaluation_data_path)
            logger.info(f"Training data shape: {train_data.shape}")

            if train_data.shape[0] < sequence_length:
                logger.error(f"Sequence length ({sequence_length}) is greater than or equal to the training or "
                             f"evaluation data size.")
                return False
            else:
                logger.info(f"Data sizes are sufficient for sequence length {sequence_length} (both training and "
                            f"evaluation).")
                return True