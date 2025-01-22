import os

import numpy as np
import tensorflow as tf
from keras.src.losses import MeanSquaredError

from cloud_node.cloud_resources_paths import CloudResourcesPaths

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


def aggregate_fog_models(received_fog_messages: dict):
    print("Running model aggregation in cloud node.")

    # load the cloud model and its weights
    custom_objects = {'mse': MeanSquaredError()}
    cloud_model_file_path = os.path.join(CloudResourcesPaths.MODELS_FOLDER_PATH,
                                         CloudResourcesPaths.CLOUD_MODEL_FILE_NAME)
    cloud_model_weights = tf.keras.models.load_model(cloud_model_file_path, custom_objects=custom_objects).get_weights()

    fog_models_data = []
    lambda_values = []
    lambda_sum = 0

    for fog_id, message in received_fog_messages:
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
        # start with the cloud model weight for the layer
        aggregated_layer = cloud_model_weights[layer_index] * 0.5  # cloud gets 50% weight

        for fog_weights, normalized_lambda in zip(fog_models_data, normalized_lambdas):
            # add the weighted contribution of the fog model's weights
            aggregated_layer += fog_weights[layer_index] * normalized_lambda * 0.5  # fog gets remaining 50%

        # update the aggregated weights
        aggregated_weights[layer_index] = aggregated_layer

    # initialize a new model with the aggregated weights
    cloud_model = tf.keras.models.load_model(cloud_model_file_path, custom_objects=custom_objects)
    cloud_model.set_weights(aggregated_weights)

    # save the updated cloud model
    cloud_model.save(cloud_model_file_path)
    print(f"Cloud model aggregation completed. New cloud model saved at: {cloud_model_file_path}")
