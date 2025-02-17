# cloudnode/model_manager.py

import os

import numpy as np
import tensorflow as tf
from keras.src.losses import MeanSquaredError

from cloud_node.cloud_resources_paths import CloudResourcesPaths
from shared.logging_config import logger


# Setting the environment variable to suppress TensorFlow low-level logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_initial_lstm_model():
    """
    Create and compile an initial LSTM model with a specified number of features and sequence length.

    Returns:
    tf.keras.Model: Compiled LSTM model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(48, 5), dtype=tf.float32),
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def aggregate_fog_models(received_fog_messages: dict):
    logger.info("Running model aggregation in cloud node.")

    custom_objects = {'mse': MeanSquaredError()}
    cloud_model_file_path = os.path.join(
        CloudResourcesPaths.MODELS_FOLDER_PATH,
        CloudResourcesPaths.CLOUD_MODEL_FILE_NAME
    )
    cloud_model_weights = tf.keras.models.load_model(cloud_model_file_path, custom_objects=custom_objects).get_weights()

    fog_models_data = []
    lambda_values = []
    lambda_sum = 0

    # Use .items() to correctly iterate over key-value pairs
    for fog_id, message in received_fog_messages.items():
        fog_model_path = message.get("fog_model_file_path")
        fog_model_weights = tf.keras.models.load_model(fog_model_path, custom_objects=custom_objects).get_weights()
        lambda_prev_value = float(message.get("lambda_prev"))
        lambda_sum += lambda_prev_value

        fog_models_data.append(fog_model_weights)
        lambda_values.append(lambda_prev_value)

    # scale lambda_prev values to normalize them in range [0, 1]
    normalized_lambdas = [lambda_prev / lambda_sum for lambda_prev in lambda_values]

    # initialize the aggregated model weights with cloud model weights
    aggregated_weights = [np.zeros_like(layer) for layer in cloud_model_weights]

    # aggregate the weights using normalized lambda values
    for layer_index in range(len(aggregated_weights)):
        # start with the cloud model weight for the layer (50% weight)
        aggregated_layer = cloud_model_weights[layer_index] * 0.5

        for fog_weights, normalized_lambda in zip(fog_models_data, normalized_lambdas):
            aggregated_layer += fog_weights[layer_index] * normalized_lambda * 0.5  # fog gets remaining 50%
        aggregated_weights[layer_index] = aggregated_layer

    # update the cloud model with the aggregated weights
    cloud_model = tf.keras.models.load_model(cloud_model_file_path, custom_objects=custom_objects)
    cloud_model.set_weights(aggregated_weights)
    cloud_model.save(cloud_model_file_path)
    logger.info(f"Cloud model aggregation completed. New cloud model saved at: {cloud_model_file_path}")

