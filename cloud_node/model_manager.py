import os
import numpy as np
import tensorflow as tf

from cloud_node.cloud_resources_paths import CloudResourcesPaths
from shared.shared_resources_paths import SharedResourcesPaths
from shared.logging_config import logger
from model_architectures import (create_initial_lstm_model, create_attention_lstm_model,
                                 create_enhanced_attention_lstm_model)

# Suppress TensorFlow low-level logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Drift detection threshold factor; fog models with drift > (threshold_factor * median) are considered outliers
DRIFT_THRESHOLD_FACTOR = 1.5


def create_model(model_type):
    if model_type == "1":
        return create_initial_lstm_model()
    elif model_type == "2":
        return create_attention_lstm_model()
    elif model_type == "3":
        return create_enhanced_attention_lstm_model()


def aggregate_fog_models(received_fog_messages: dict):
    logger.info("Running model aggregation in cloud node with drift detection and robust correction.")

    custom_objects = {
        "LogCosh": tf.keras.losses.LogCosh(),
        "mse": tf.keras.losses.MeanSquaredError(),
        "Huber": tf.keras.losses.Huber()
    }
    cloud_model_file_path = os.path.join(
        CloudResourcesPaths.MODELS_FOLDER_PATH,
        CloudResourcesPaths.CLOUD_MODEL_FILE_NAME
    )

    cache_cloud_model_file_path = os.path.join(SharedResourcesPaths.CACHE_FOLDER_PATH,
                                               CloudResourcesPaths.CLOUD_MODEL_FILE_NAME)

    # Load current cloud model weights; if not present in the main path, use the cache
    if os.path.exists(cloud_model_file_path):
        cloud_model = tf.keras.models.load_model(cloud_model_file_path, custom_objects=custom_objects)
    else:
        cloud_model = tf.keras.models.load_model(cache_cloud_model_file_path, custom_objects=custom_objects)
    cloud_model_weights = cloud_model.get_weights()

    fog_models_data = []  # each element is a list of weights per layer for a fog model
    lambda_values = []    # cumulative performance factors (λ) for each fog

    # Iterate over fog messages and load their model weights and lambda values
    for fog_id, message in received_fog_messages.items():
        fog_model_path = message.get("fog_model_file_path")
        fog_model = tf.keras.models.load_model(fog_model_path, custom_objects=custom_objects)
        fog_model_weights = fog_model.get_weights()
        lambda_prev_value = float(message.get("lambda_prev"))
        fog_models_data.append(fog_model_weights)
        lambda_values.append(lambda_prev_value)

    # Normalize lambda values to sum to 1 (for weighting)
    lambda_sum = sum(lambda_values)
    normalized_lambdas = [lmbd / lambda_sum for lmbd in lambda_values] if lambda_sum != 0 else [0 for _ in
                                                                                                lambda_values]

    # Initialize aggregated weights (layer-wise aggregation)
    aggregated_weights = [np.zeros_like(layer) for layer in cloud_model_weights]

    # For each layer, perform robust aggregation:
    for layer_index in range(len(aggregated_weights)):
        cloud_layer = cloud_model_weights[layer_index]

        # Compute drift (L2 norm difference) for each fog's layer weight from the cloud's layer weight
        fog_drifts = []
        for fog_weights in fog_models_data:
            fog_layer = fog_weights[layer_index]
            drift = np.linalg.norm(fog_layer - cloud_layer)
            fog_drifts.append(drift)
        # Compute median drift for the current layer
        median_drift = np.median(fog_drifts)

        # Separate good and outlier (bad performance) fog models based on drift threshold
        good_fog_weights = []
        good_fog_lambdas = []
        for idx, drift in enumerate(fog_drifts):
            if drift <= DRIFT_THRESHOLD_FACTOR * median_drift:
                good_fog_weights.append(fog_models_data[idx][layer_index])
                good_fog_lambdas.append(normalized_lambdas[idx])
            else:
                logger.warning(f"Fog model index {idx} in layer {layer_index} excluded due to high drift "
                               f"(drift={drift:.4f}, median={median_drift:.4f}).")

        # If there are no good fog models for this layer, preserve the cloud model weights
        if not good_fog_weights:
            logger.warning(f"No good fog models detected in layer {layer_index}; preserving cloud model weights for "
                           f"this layer.")
            aggregated_weights[layer_index] = cloud_layer
            continue

        # Otherwise, aggregate fog contributions for this layer:
        # Start with the cloud weight weighted at 50%
        aggregated_layer = cloud_layer * 0.5

        # Compute weighted average using normalized lambda values from the good fog models.
        # If lambdas sum to zero (edge case), fallback to median aggregation.
        if sum(good_fog_lambdas) == 0:
            fog_aggregate = np.median(good_fog_weights, axis=0)
        else:
            weighted_sum = np.zeros_like(cloud_layer)
            total_lambda = sum(good_fog_lambdas)
            for w, lam in zip(good_fog_weights, good_fog_lambdas):
                weighted_sum += w * lam
            fog_aggregate = weighted_sum / total_lambda

        aggregated_layer += fog_aggregate * 0.5  # fog models contribute the remaining 50%
        aggregated_weights[layer_index] = aggregated_layer

    # Update the cloud model with the new aggregated weights and save
    cloud_model.set_weights(aggregated_weights)
    cloud_model.save(cloud_model_file_path)
    cloud_model.save(cache_cloud_model_file_path)

    logger.info(f"Cloud model aggregation with drift correction completed. New cloud model saved at: "
                f"{cloud_model_file_path} and cached.")
