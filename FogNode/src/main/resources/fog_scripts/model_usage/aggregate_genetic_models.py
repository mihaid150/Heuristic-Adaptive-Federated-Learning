import sys
import os
import logging
import warnings
import traceback
import json
import random
import numpy as np
import tensorflow as tf
import scipy.constants as sc
from tensorflow.keras.losses import MeanSquaredError

logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Compiled the loaded model, but the compiled metrics have yet to be built")

LAMBDA_FILE = '/app/models/lambda_prev.json'


def save_lambda_prev(lambda_prev):
    """
    Saves the current value of `lambda_prev` to a JSON file.

    Args:
        lambda_prev (float): The current value of lambda to be saved.
    """
    print(f"Saving lambda_prev: {lambda_prev} to {LAMBDA_FILE}")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(LAMBDA_FILE), exist_ok=True)

    data = {'lambda_prev': lambda_prev}
    with open(LAMBDA_FILE, 'w') as file:
        json.dump(data, file)


def read_lambda_prev():
    """
    Reads the value of `lambda_prev` from a JSON file. Returns a default value if the file does not exist.

    Returns:
        float: The value of `lambda_prev`, defaulting to 1 if the file does not exist.
    """
    if os.path.exists(LAMBDA_FILE):
        with open(LAMBDA_FILE, 'r') as file:
            data = json.load(file)
            lambda_prev = data.get('lambda_prev', 0)
            print(f"Read lambda_prev: {lambda_prev} from {LAMBDA_FILE}")
            return lambda_prev
    print(f"{LAMBDA_FILE} does not exist. Returning default lambda_prev: 1")
    return 1


def init_new_model(model_weights):
    """
    Initializes a new model with weights set to zeros.

    Args:
        model_weights (list): The weights of an existing model.

    Returns:
        list: A list of arrays initialized to zero with the same shape as the input weights.
    """
    print("Initializing new model with zeros")
    return [np.zeros_like(w) for w in model_weights]


def aggregation(edge_model_weights, fog_model_path_weights, mu_new, mu_prev, fog_model_path, lambda_prev,
                custom_objects):
    """
    Aggregates the edge and fog models using weighted averages and updates the fog model.

    Args:
        edge_model_weights (list): Weights of the edge model.
        fog_model_path_weights (list): Weights of the fog model.
        mu_new (float): Weighting factor for the new edge model.
        mu_prev (float): Weighting factor for the previous fog model.
        fog_model_path (str): Path to the fog model file.
        lambda_prev (float): Previous lambda value.
        custom_objects (dict): Custom objects used for model loading.
    """
    print("Starting aggregation of models")
    aggregated_model_weights = init_new_model(fog_model_path_weights)

    for i in range(len(aggregated_model_weights)):
        aggregated_model_weights[i] = ((mu_new * edge_model_weights[i] +
                                        mu_prev * fog_model_path_weights[i]) /
                                       (mu_new + mu_prev))
    print("Aggregated model weights calculated")

    new_fog_model = tf.keras.models.load_model(fog_model_path, custom_objects=custom_objects)
    new_fog_model.set_weights(aggregated_model_weights)
    new_fog_model.save(fog_model_path)
    print(f"Aggregated model saved to {fog_model_path}")
    lambda_prev += mu_new
    save_lambda_prev(lambda_prev)


def aggregate_models(edge_model_path, new_edge_perf, fog_model_path, old_fog_perf, current_temperature,
                     init_temperature):
    """
    Aggregates models from the edge and fog nodes, considering their performance and a probabilistic factor.

    Args:
        edge_model_path (str): Path to the edge model file.
        new_edge_perf (float): Performance metric of the new edge model.
        fog_model_path (str): Path to the fog model file.
        old_fog_perf (float): Performance metric of the current fog model.
        current_temperature (float): Current temperature for simulated annealing.
        init_temperature (float): Initial temperature for simulated annealing.
    """
    try:
        print(f"Loading models from {edge_model_path} and {fog_model_path}")
        custom_objects = {'mse': MeanSquaredError()}
        edge_model_weights = tf.keras.models.load_model(edge_model_path, custom_objects=custom_objects).get_weights()
        fog_model_weights = tf.keras.models.load_model(fog_model_path, custom_objects=custom_objects).get_weights()

        lambda_prev = read_lambda_prev()

        mu_new = np.exp(current_temperature / init_temperature)
        mu_prev = np.exp(lambda_prev / init_temperature)

        print(f"new_edge_perf: {new_edge_perf}, old_fog_perf: {old_fog_perf}")
        print(f"mu_new: {mu_new}, mu_prev: {mu_prev}, lambda_prev: {lambda_prev}")

        if new_edge_perf < old_fog_perf:
            print("New edge performance is better. Aggregating models.")
            aggregation(edge_model_weights, fog_model_weights, mu_new, mu_prev, fog_model_path, lambda_prev,
                        custom_objects)
        else:
            delta = new_edge_perf - old_fog_perf
            gamma = random.random()
            factor = (-delta) / (sc.Boltzmann * init_temperature)
            print(f"delta: {delta}, gamma: {gamma}, factor: {factor}")

            if gamma > factor:
                print("Aggregating models based on probability factor.")
                aggregation(edge_model_weights, fog_model_weights, mu_new, mu_prev, fog_model_path, lambda_prev,
                            custom_objects)
            else:
                print("Keeping the existing fog model.")
                new_fog_model = tf.keras.models.load_model(fog_model_path, custom_objects=custom_objects)
                new_fog_model.set_weights(fog_model_weights)
                new_fog_model.save(fog_model_path)
    except Exception as e:
        error = {'error from aggregate_models': str(e)}
        print(json.dumps(error), file=sys.stderr)
        traceback.print_exc()


if __name__ == '__main__':
    """
    Entry point for the script. Expects the following command-line arguments:
        1. edge_model_path (str): Path to the edge model file.
        2. new_edge_perf (float): Performance metric of the new edge model.
        3. fog_model_path (str): Path to the fog model file.
        4. old_fog_perf (float): Performance metric of the current fog model.
        5. current_temperature (float): Current temperature for simulated annealing.
        6. init_temperature (float): Initial temperature for simulated annealing.
    """
    path_edge_model = sys.argv[1]
    perf_new_edge = float(sys.argv[2])
    path_fog_model = sys.argv[3]
    perf_old_fog = float(sys.argv[4])
    curr_temperature = float(sys.argv[5])
    max_temperature = float(sys.argv[6])
    print(f"Starting model aggregation with arguments: {sys.argv[1:]}")
    aggregate_models(path_edge_model, perf_new_edge, path_fog_model, perf_old_fog, curr_temperature, max_temperature)
