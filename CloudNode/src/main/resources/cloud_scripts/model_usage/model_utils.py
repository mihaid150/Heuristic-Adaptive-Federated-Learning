import logging
import warnings
import traceback
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import os
import sys

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Compiled the loaded model, but the compiled metrics have yet to be built")
logging.basicConfig(level=logging.INFO)


def init_new_model(model_weights):
    """
    Initialize a new model with zero weights, matching the structure of the provided weights.

    Args:
        model_weights (list): List of NumPy arrays representing the model's weights.

    Returns:
        list: List of NumPy arrays initialized with zeros, matching the input weights' structure.
    """
    return [np.zeros_like(w) for w in model_weights]


def load_model_weights(model_path):
    """
    Load the weights and layers of a Keras model from the specified file path.

    Args:
        model_path (str): Path to the Keras model file.

    Returns:
        tuple: A tuple containing:
            - List of NumPy arrays representing the model's weights.
            - List of Keras layers in the model.

    Raises:
        OSError: If the model file does not exist or cannot be loaded.
    """
    custom_objects = {'mse': MeanSquaredError()}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model.get_weights(), model.layers


def validate_model_shapes(global_model_weights, fog_models_weights):
    """
    Validate that the weights of the global model and fog models have matching shapes.

    Args:
        global_model_weights (list): List of NumPy arrays representing the global model's weights.
        fog_models_weights (list): List of lists of NumPy arrays representing fog models' weights.

    Raises:
        ValueError: If the number of layers or the shape of any layer's weights do not match.
    """
    for local_weights in fog_models_weights:
        if len(global_model_weights) != len(local_weights):
            raise ValueError(f"Model shapes do not match: global model has {len(global_model_weights)} "
                             f"layers, but a local model has {len(local_weights)} layers")

    for i, (global_weight, local_weight) in enumerate(zip(global_model_weights, fog_models_weights[0])):
        if global_weight.shape != local_weight.shape:
            raise ValueError(f"Layer {i} shapes do not match: global model layer shape is {global_weight.shape}, "
                             f"but a local model layer shape is {local_weight.shape}")


def save_aggregated_model(global_model_path, aggregated_model_weights):
    """
    Save the aggregated model weights to the specified global model file.

    Args:
        global_model_path (str): Path to the global model file.
        aggregated_model_weights (list): List of NumPy arrays representing the aggregated weights.

    Returns:
        None

    Raises:
        OSError: If the model file cannot be saved.
    """
    custom_objects = {'mse': MeanSquaredError()}
    global_model = tf.keras.models.load_model(global_model_path, custom_objects=custom_objects)
    global_model.set_weights(aggregated_model_weights)
    global_model.compile(optimizer=Adam(), loss='mse', metrics=['mse'])
    global_model.save(global_model_path)
    print(f"Aggregated model saved at {global_model_path}")


def log_error(error_message):
    """
    Log an error message and print the traceback to standard error.

    Args:
        error_message (str): The error message to log.

    Returns:
        None
    """
    error = {'error': error_message}
    print(json.dumps(error), file=sys.stderr)
    traceback.print_exc()
