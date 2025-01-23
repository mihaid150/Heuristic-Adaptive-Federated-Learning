import os
import json
import random
import scipy.constants as sc
import numpy as np
import tensorflow as tf
from keras.src.losses import MeanSquaredError

from fog_resources_paths import FogResourcesPaths
from fog_cooling_scheduler import FogCoolingScheduler


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
        "loss": 0.4,  # higher weight for loss
        "mae": 0.3,  # medium weight for mae
        "mse": 0.1,  # lower weight for mse since it's redundant loss
        "rmse": 0.1,  # lower weight for rmse
        "r2": -0.1,  # negative weight because higher r2 is better
    }
    for metric, weight in weights.items():
        score += weight * metrics_for_score[metric]
    return score


def execute_models_aggregation(fog_cooling_scheduler: FogCoolingScheduler, metrics):
    print("Running the model aggregation in the fog node.")

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

    mu_new = np.exp(fog_cooling_scheduler.temperature / fog_cooling_scheduler.initial_temperature)
    mu_prev = np.exp(lambda_prev / fog_cooling_scheduler.initial_temperature)

    before_training_score = compute_weighted_score(metrics["before_training"])
    after_training_score = compute_weighted_score(metrics["after_training"])

    if after_training_score < before_training_score:
        print("Performance score of the after training model is better. Proceeding with aggregation.")
        aggregate_models(edge_model_weights, fog_model_weights, mu_new, mu_prev, lambda_prev)
    else:
        delta = after_training_score - before_training_score
        scaled_boltzmann_constant = sc.Boltzmann * 1e23

        normalized_temp = fog_cooling_scheduler.temperature / max(fog_cooling_scheduler.initial_temperature, 1)

        gamma = random.random()
        factor = np.exp(-delta / (scaled_boltzmann_constant * normalized_temp))

        if gamma > factor:
            print("Aggregating models based on the probability factor.")
            aggregate_models(edge_model_weights, fog_model_weights, mu_new, mu_prev, lambda_prev)
        else:
            print("Keeping the current fog model.")


def softmax(values):
    values = np.array(values)
    exp_values = np.exp(values - np.max(values))
    return exp_values / np.sum(exp_values)


def aggregate_models(edge_model_weights, fog_model_weights, mu_new, mu_prev, lambda_prev):
    weights = softmax([mu_new, mu_prev])
    mu_new_normalized = weights[0]
    mu_prev_normalized = weights[1]

    print(f"Normalized weights -> mu_new: {mu_new_normalized}, mu_prev: {mu_prev_normalized}")

    # perform weighted aggregation of the models
    new_weights = []
    for edge_layer, fog_layer in zip(edge_model_weights, fog_model_weights):
        aggregated_layer = ((mu_new_normalized * edge_layer + mu_prev_normalized * fog_layer) /
                            (mu_new_normalized + mu_prev_normalized))
        new_weights.append(aggregated_layer)

    # update fog model with aggregated weights
    fog_model_weights.set_weights(new_weights)

    # save the updated fog model
    fog_model_file_path = os.path.join(
        FogResourcesPaths.MODELS_FOLDER_PATH,
        FogResourcesPaths.FOG_MODEL_FILE_NAME
    )
    fog_model_weights.save(fog_model_file_path)
    print(f"Aggregated model saved at: {fog_model_file_path}")

    # save the updated lambda_prev value
    lambda_prev += mu_prev
    save_lambda_prev(lambda_prev)
    print(f"Updated lambda_prev value saved: {lambda_prev}")
