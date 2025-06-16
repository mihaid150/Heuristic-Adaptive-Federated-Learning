# fog_node/model_manager.py

import os
import json
import random
import scipy.constants as sc
import numpy as np
import tensorflow as tf
from keras.src.losses import MeanSquaredError

from shared.logging_config import logger
from fog_node.fog_resources_paths import FogResourcesPaths
from fog_node.fog_cooling_scheduler import FogCoolingScheduler
from shared.utils import metric_weights


def save_lambda_prev(lambda_prev) -> None:
    data = {'lambda_prev': lambda_prev}

    lambda_prev_file_path = os.path.join(
        FogResourcesPaths.MODELS_FOLDER_PATH,
        FogResourcesPaths.LAMBDA_PREV_FILE_NAME
    )

    with open(lambda_prev_file_path, 'w') as file:
        json.dump(data, file)


def read_lambda_prev():
    lambda_prev_file_path = os.path.join(
        FogResourcesPaths.MODELS_FOLDER_PATH,
        FogResourcesPaths.LAMBDA_PREV_FILE_NAME
    )
    if os.path.exists(lambda_prev_file_path):
        with open(lambda_prev_file_path, 'r') as file:
            data = json.load(file)
            lambda_prev = data.get('lambda_prev', 0)
            return lambda_prev
    return 1


def compute_weighted_score(metrics_for_score):
    score = 0
    for metric, weight in metric_weights.items():
        score += weight * metrics_for_score[metric]
    return score


def logistic(x, l=1, k=1, x0=0):
    """Computes the logistic function: l / (1 + exp(-k*(x-x0)))."""
    return l / (1 + np.exp(-k * (x - x0)))


def detect_concept_drift(before_training_score, after_training_score, drift_threshold=0.5):
    """
    Simple detection: if performance degrades beyond a threshold, assume drift.
    """
    performance_delta = after_training_score - before_training_score
    return performance_delta > drift_threshold


def compute_adaptive_weights(mu_new, mu_prev, recent_performance_factor, momentum=0.9, previous_adaptive_weights=None):
    """
    Adjust the new factor based on recent performance improvement.
    Optionally smooth the weights using momentum.
    If previous_adaptive_weights is provided as [prev_mu_new, prev_mu_prev], they are used for smoothing.
    """
    adjusted_mu_new = mu_new * recent_performance_factor
    raw_weights = [adjusted_mu_new, mu_prev]
    logger.info(f"Raw adaptive weights before smoothing: adjusted_mu_new: {adjusted_mu_new}, mu_prev: {mu_prev}")

    # If previous weights exist, smooth them
    if previous_adaptive_weights is not None:
        smoothed_weights = [
            momentum * prev + (1 - momentum) * current
            for prev, current in zip(previous_adaptive_weights, raw_weights)
        ]
        logger.info(f"Smoothed adaptive weights: {smoothed_weights}")
    else:
        smoothed_weights = raw_weights

    # Normalize using softmax to sum to 1
    normalized_weights = softmax(smoothed_weights)
    return normalized_weights


def execute_models_aggregation(fog_cooling_scheduler: FogCoolingScheduler, metrics):
    logger.info("Running the model aggregation in the fog node.")

    custom_objects = {
        "LogCosh": tf.keras.losses.LogCosh(),
        "mse": tf.keras.losses.MeanSquaredError(),
        "Huber": tf.keras.losses.Huber()
    }

    edge_model_file_path = os.path.join(
        FogResourcesPaths.MODELS_FOLDER_PATH,
        FogResourcesPaths.EDGE_MODEL_FILE_NAME
    )
    fog_model_file_path = os.path.join(
        FogResourcesPaths.MODELS_FOLDER_PATH,
        FogResourcesPaths.FOG_MODEL_FILE_NAME
    )

    edge_model_weights = tf.keras.models.load_model(edge_model_file_path, custom_objects=custom_objects).get_weights()
    fog_model_weights = tf.keras.models.load_model(fog_model_file_path, custom_objects=custom_objects).get_weights()

    lambda_prev = read_lambda_prev()

    before_training_score = compute_weighted_score(metrics["before_training"])
    after_training_score = compute_weighted_score(metrics["after_training"])
    spike_intensity = metrics["spike_intensity"]

    mu_new = logistic(fog_cooling_scheduler.temperature / fog_cooling_scheduler.initial_temperature, l=1, k=1, x0=0)
    intensity_factor = np.log1p(spike_intensity)
    logger.info(f"Spike intensity factor: {intensity_factor:.4f}; before mu_new: {mu_new:.4f}")
    mu_new *= intensity_factor
    logger.info(f"Spike intensity factor: {intensity_factor:.4f}; after mu_new: {mu_new:.4f}")

    mu_prev = logistic(lambda_prev / fog_cooling_scheduler.initial_temperature, l=1, k=1, x0=0)

    recent_performance_factor = 1.0
    if detect_concept_drift(before_training_score, after_training_score):
        logger.info("Concept drift detected. Adjusting aggregation weights.")
        lambda_prev = 0
        mu_prev = logistic(lambda_prev / fog_cooling_scheduler.initial_temperature, l=1, k=1, x0=0)
        recent_performance_factor = 1.5

    if after_training_score < before_training_score:
        logger.info("After-training model performance improved. Proceeding with aggregation.")
    else:
        delta = after_training_score - before_training_score
        scaled_boltzmann_constant = sc.Boltzmann * 1e23
        normalized_temp = fog_cooling_scheduler.temperature / max(fog_cooling_scheduler.initial_temperature, 1)
        gamma = random.random()
        factor = np.exp(-delta / (scaled_boltzmann_constant * normalized_temp))
        if gamma > factor:
            logger.info("Aggregating models based on the probability factor.")
        else:
            logger.info("Keeping the current fog model.")
            return

    # Compute adaptive weights (using momentum if desired)
    adaptive_weights = compute_adaptive_weights(mu_new, mu_prev, recent_performance_factor, momentum=0.9)
    mu_new_normalized, mu_prev_normalized = adaptive_weights
    logger.info(
        f"Final adaptive weights: mu_new_normalized: {mu_new_normalized}, mu_prev_normalized: {mu_prev_normalized}")

    # Perform aggregation layer-wise with additional robust check
    new_weights = []
    for i, (edge_layer, fog_layer) in enumerate(zip(edge_model_weights, fog_model_weights)):
        # Log L2 norms before aggregation
        edge_norm = np.linalg.norm(edge_layer)
        fog_norm = np.linalg.norm(fog_layer)
        logger.info(f"Layer {i} - Edge L2 norm: {edge_norm:.4f}, Fog L2 norm: {fog_norm:.4f}")

        # Compute the weighted average candidate
        weighted_avg = ((mu_new_normalized * edge_layer + mu_prev_normalized * fog_layer) /
                        (mu_new_normalized + mu_prev_normalized))

        # Robust aggregation: if the relative difference between edge and fog is large,
        # use the median (or trimmed mean) instead.
        diff_norm = np.linalg.norm(edge_layer - fog_layer)
        mean_norm = (edge_norm + fog_norm) / 2.0
        if mean_norm > 0 and (diff_norm / mean_norm) > 0.5:  # threshold can be tuned
            robust_value = np.median(np.stack([edge_layer, fog_layer], axis=0), axis=0)
            logger.info(f"Layer {i} using robust aggregation (median) due to high relative difference.")
            aggregated_layer = robust_value
        else:
            aggregated_layer = weighted_avg

        new_weights.append(aggregated_layer)
        # Log L2 norm after aggregation for this layer
        agg_norm = np.linalg.norm(aggregated_layer)
        logger.info(f"Layer {i} - Aggregated L2 norm: {agg_norm:.4f}")

    custom_objects = {'mse': MeanSquaredError()}
    fog_model = tf.keras.models.load_model(fog_model_file_path, custom_objects=custom_objects)
    fog_model.set_weights(new_weights)
    fog_model.save(fog_model_file_path)
    logger.info(f"Aggregated model saved at: {fog_model_file_path}")

    decay_factor = 0.9
    max_lambda = 100.0
    if after_training_score < before_training_score:
        lambda_prev = 0.5 * lambda_prev
    else:
        lambda_prev = decay_factor * lambda_prev + (1 - decay_factor) * mu_prev
    lambda_prev = min(lambda_prev, max_lambda)
    save_lambda_prev(lambda_prev)
    logger.info(f"Updated lambda_prev value saved: {lambda_prev}")


def softmax(values):
    values = np.array(values)
    exp_values = np.exp(values - np.max(values))
    return exp_values / np.sum(exp_values)
