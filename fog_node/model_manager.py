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
    weights = {
        "loss": 0.1,  # higher weight for loss
        "mae": 0.3,  # medium weight for mae
        "mse": 0.1,  # lower weight for mse since it's redundant loss
        "rmse": 0.1,  # lower weight for rmse
        "r2": -0.2,  # negative weight because higher r2 is better
    }
    for metric, weight in weights.items():
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


def compute_adaptive_weights(mu_new, mu_prev, recent_performance_factor):
    """
    Adjust the new factor based on recent performance improvement.
    E.g., if recent_performance_factor > 1, it boosts μ_new.
    """
    adjusted_mu_new = mu_new * recent_performance_factor
    return softmax([adjusted_mu_new, mu_prev])


def execute_models_aggregation(fog_cooling_scheduler: FogCoolingScheduler, metrics):
    logger.info("Running the model aggregation in the fog node.")

    custom_objects = {'mse': MeanSquaredError()}

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

    # Compute performance scores
    before_training_score = compute_weighted_score(metrics["before_training"])
    after_training_score = compute_weighted_score(metrics["after_training"])

    # Use logistic function to compute scaling factors.
    # Dividing by initial_temperature normalizes the input.
    mu_new = logistic(fog_cooling_scheduler.temperature / fog_cooling_scheduler.initial_temperature, l=1, k=1, x0=0)
    mu_prev = logistic(lambda_prev / fog_cooling_scheduler.initial_temperature, l=1, k=1, x0=0)

    # Check for concept drift
    recent_performance_factor = 1.0
    if detect_concept_drift(before_training_score, after_training_score):
        logger.info("Concept drift detected. Adjusting aggregation weights.")
        lambda_prev = 0  # Reset historical factor to reduce its influence
        mu_prev = logistic(lambda_prev / fog_cooling_scheduler.initial_temperature, l=1, k=1, x0=0)
        recent_performance_factor = 1.5  # Boost influence of the new model

    # Decide if aggregation should proceed based on performance improvement
    if after_training_score < before_training_score:
        logger.info("Performance score of the after training model is better. Proceeding with aggregation.")
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

    # Perform aggregation using adaptive weights and pass performance scores for lambda_prev update.
    aggregate_models(edge_model_weights, fog_model_weights, mu_new, mu_prev, lambda_prev, recent_performance_factor,
                     before_training_score, after_training_score)


def softmax(values):
    values = np.array(values)
    exp_values = np.exp(values - np.max(values))
    return exp_values / np.sum(exp_values)


def aggregate_models(edge_model_weights, fog_model_weights_list, mu_new, mu_prev, lambda_prev,
                     recent_performance_factor, before_training_score, after_training_score):
    # Compute adaptive weights using the new factor boost if needed
    weights = compute_adaptive_weights(mu_new, mu_prev, recent_performance_factor)
    mu_new_normalized = weights[0]
    mu_prev_normalized = weights[1]

    logger.info(f"Normalized weights -> mu_new: {mu_new_normalized}, mu_prev: {mu_prev_normalized}")

    # Perform weighted aggregation of the models layer-wise
    new_weights = []
    for edge_layer, fog_layer in zip(edge_model_weights, fog_model_weights_list):
        aggregated_layer = ((mu_new_normalized * edge_layer + mu_prev_normalized * fog_layer) /
                            (mu_new_normalized + mu_prev_normalized))
        new_weights.append(aggregated_layer)

    # Update the fog model with the aggregated weights
    custom_objects = {'mse': MeanSquaredError()}
    fog_model = tf.keras.models.load_model(
        os.path.join(FogResourcesPaths.MODELS_FOLDER_PATH, FogResourcesPaths.FOG_MODEL_FILE_NAME),
        custom_objects=custom_objects
    )
    fog_model.set_weights(new_weights)
    fog_model.save(os.path.join(FogResourcesPaths.MODELS_FOLDER_PATH, FogResourcesPaths.FOG_MODEL_FILE_NAME))
    fog_model_path = os.path.join(FogResourcesPaths.MODELS_FOLDER_PATH, FogResourcesPaths.FOG_MODEL_FILE_NAME)
    logger.info(f"Aggregated model saved at: {fog_model_path}")

    # Update λ_prev based on performance improvement
    decay_factor = 0.9
    max_lambda = 100.0
    if after_training_score < before_training_score:
        # Good improvement: reduce historical influence
        lambda_prev = 0.5 * lambda_prev
    else:
        # Weighted update to gradually incorporate historical performance
        lambda_prev = decay_factor * lambda_prev + (1 - decay_factor) * mu_prev

    lambda_prev = min(lambda_prev, max_lambda)
    save_lambda_prev(lambda_prev)
    logger.info(f"Updated lambda_prev value saved: {lambda_prev}")
