import os
import pandas as pd
import tensorflow as tf
import numpy as np

from data_selection import filter_data_by_interval_date, filter_data_by_day_date
from edge_resources_paths import EdgeResourcesPaths
from data_preprocessing import preprocess_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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


def pretrain_edge_model(edge_model_file_path: str, start_date: str, end_date: str, learning_rate: float,
                        batch_size: int, epochs: int, patience: int, fine_tune_layers: int):
    """
    Pretrain the model on data from a given period and evaluate on the same period.
    Save evaluation metrics before and after training in a JSON file.

    :param edge_model_file_path: Path to the saved model file.
    :param start_date: The start date for the training period.
    :param end_date: The end date for the training period.
    :param learning_rate: Learning rate for training.
    :param batch_size: Batch size for training.
    :param epochs: Number of epochs for training.
    :param patience: Early stopping patience for training.
    :param fine_tune_layers: Number of layers to fine-tune during training.
    """
    # Load data for the specified period
    filter_data_by_interval_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime", start_date, end_date,
                                 EdgeResourcesPaths.FILTERED_DATA_PATH)
    preprocess_data(EdgeResourcesPaths.FILTERED_DATA_PATH, "datetime", "value")

    data = pd.read_csv(EdgeResourcesPaths.FILTERED_DATA_PATH)

    # Prepare input and target data
    X = data[[
        "value_rolling_mean_3", "value_rolling_max_3", "value_rolling_min_3",
        "value_rolling_mean_6", "value_rolling_max_6"
    ]].values
    y = data["value"].values

    # Split data into training and validation sets
    split_index = int(len(X) * 0.8)
    train_X, val_X = X[:split_index], X[split_index:]
    train_y, val_y = y[:split_index], y[split_index:]

    # Load the model
    custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
    model = tf.keras.models.load_model(edge_model_file_path, custom_objects=custom_objects)

    # Evaluate before training
    print("Evaluating the model before training...")
    predictions_before = model.predict(val_X)
    evaluation_before = {
        "loss": float("inf"),  # Large max value since the model is untrained
        "mae": float("inf"),
        "mse": float("inf"),
        "rmse": float("inf"),
        "r2": -float("inf")  # Minimum possible R2 score
    }
    try:
        evaluation_before.update({
            "loss": mean_squared_error(val_y, predictions_before),
            "mae": mean_absolute_error(val_y, predictions_before),
            "mse": mean_squared_error(val_y, predictions_before),
            "rmse": np.sqrt(mean_squared_error(val_y, predictions_before)),
            "r2": r2_score(val_y, predictions_before)
        })
    except Exception:
        print("Evaluation before training failed, possibly due to untrained model.")

    # Fine-tune the model (only specific layers)
    for layer in model.layers[:-fine_tune_layers]:
        layer.trainable = False

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=["mae", "mse"])

    # Train the model
    print("Training the model...")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate after training
    print("Evaluating the model after training...")
    predictions_after = model.predict(val_X)
    evaluation_after = {
        "loss": mean_squared_error(val_y, predictions_after),
        "mae": mean_absolute_error(val_y, predictions_after),
        "mse": mean_squared_error(val_y, predictions_after),
        "rmse": np.sqrt(mean_squared_error(val_y, predictions_after)),
        "r2": r2_score(val_y, predictions_after)
    }

    # Save metrics to a JSON file
    metrics = {
        "before_training": evaluation_before,
        "after_training": evaluation_after
    }

    # Save the pretrained model
    pretrained_model_file_path = os.path.join(EdgeResourcesPaths.MODELS_FOLDER_PATH,
                                              EdgeResourcesPaths.RETRAINED_EDGE_MODEL_FILE_NAME)
    model.save(pretrained_model_file_path)
    print(f"Pretrained model saved at: {pretrained_model_file_path}")

    return metrics


def retrain_edge_model(edge_model_file_path: str, date: str, learning_rate: float,
                       batch_size: int, epochs: int, patience: int, fine_tune_layers: int):
    """
    Retrain the model on the current day's data and evaluate it on the next day's data.
    Save evaluation metrics before and after retraining in a JSON file.

    :param edge_model_file_path: Path to the saved model file.
    :param date: The date for the current day (training day).
    :param learning_rate: Learning rate for retraining.
    :param batch_size: Batch size for retraining.
    :param epochs: Number of epochs for retraining.
    :param patience: Early stopping patience for retraining.
    :param fine_tune_layers: Number of layers to fine-tune during retraining.
    """
    # Paths for data
    current_day_data_path = EdgeResourcesPaths.CURRENT_DAY_DATA_PATH
    next_day_data_path = EdgeResourcesPaths.NEXT_DAY_DATA_PATH

    # Load data for the current day and the next day
    filter_data_by_day_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime", date, current_day_data_path)
    preprocess_data(current_day_data_path, "datetime", "value")

    next_day = pd.to_datetime(date) + pd.Timedelta(days=1)
    filter_data_by_day_date(EdgeResourcesPaths.INPUT_DATA_PATH, "datetime", next_day.strftime("%Y-%m-%d"), next_day_data_path)
    preprocess_data(next_day_data_path, "datetime", "value")

    train_data = pd.read_csv(current_day_data_path)
    eval_data = pd.read_csv(next_day_data_path)

    # Prepare input and target data
    train_X = train_data[[
        "value_rolling_mean_3", "value_rolling_max_3", "value_rolling_min_3",
        "value_rolling_mean_6", "value_rolling_max_6"
    ]].values
    train_y = train_data["value"].values

    eval_X = eval_data[[
        "value_rolling_mean_3", "value_rolling_max_3", "value_rolling_min_3",
        "value_rolling_mean_6", "value_rolling_max_6"
    ]].values
    eval_y = eval_data["value"].values

    # Load the model
    custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
    model = tf.keras.models.load_model(edge_model_file_path, custom_objects=custom_objects)

    # Evaluate before retraining
    print("Evaluating the model before retraining...")
    evaluation_before = model.evaluate(eval_X, eval_y, batch_size=batch_size, verbose=1)
    predictions_before = model.predict(eval_X)
    r2_before = r2_score(eval_y, predictions_before)
    rmse_before = np.sqrt(evaluation_before[1])

    metrics_before = {
        "loss": evaluation_before[0],
        "mae": evaluation_before[1],
        "mse": evaluation_before[2],
        "rmse": rmse_before,
        "r2": r2_before
    }

    # Fine-tune the model (only specific layers)
    for layer in model.layers[:-fine_tune_layers]:
        layer.trainable = False

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=["mae", "mse"])

    # Retrain the model
    print("Retraining the model...")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    history = model.fit(
        train_X, train_y,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate after retraining
    print("Evaluating the model after retraining...")
    evaluation_after = model.evaluate(eval_X, eval_y, batch_size=batch_size, verbose=1)
    predictions_after = model.predict(eval_X)
    r2_after = r2_score(eval_y, predictions_after)
    rmse_after = np.sqrt(evaluation_after[1])

    metrics_after = {
        "loss": evaluation_after[0],
        "mae": evaluation_after[1],
        "mse": evaluation_after[2],
        "rmse": rmse_after,
        "r2": r2_after
    }

    # Save metrics to a JSON file
    metrics = {
        "before_training": metrics_before,
        "after_training": metrics_after
    }

    # Save the retrained model
    retrained_edge_model_file_path = os.path.join(EdgeResourcesPaths.MODELS_FOLDER_PATH,
                                                  EdgeResourcesPaths.RETRAINED_EDGE_MODEL_FILE_NAME)
    model.save(retrained_edge_model_file_path)
    print(f"Retrained model saved at: {retrained_edge_model_file_path}")

    return metrics
