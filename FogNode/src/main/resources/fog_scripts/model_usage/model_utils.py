import sys
import logging
import warnings
import traceback
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam


# Configure logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Compiled the loaded model, but the compiled metrics have yet to be built")
logging.basicConfig(level=logging.INFO)


def init_new_model(model_weights):
    """
    Initializes a new model with zeroed weights based on the shape of an existing model's weights.

    Parameters:
    - model_weights (list[np.ndarray]): Weights of an existing model.

    Returns:
    - list[np.ndarray]: A list of zero-initialized weights with the same shape as the input weights.
    """
    return [np.zeros_like(w) for w in model_weights]


def load_model_weights(model_path):
    """
    Loads the weights of a TensorFlow/Keras model from the specified path.

    Parameters:
    - model_path (str): Path to the saved model.

    Returns:
    - list[np.ndarray]: A list of weights of the loaded model.
    """
    custom_objects = {'mse': MeanSquaredError()}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects).get_weights()


def validate_model_shapes(fog_model_weights, edge_models_weights):
    """
    Validates that the shapes of the fog model and edge models are compatible for aggregation.

    Parameters:
    - fog_model_weights (list[np.ndarray]): Weights of the fog model.
    - edge_models_weights (list[list[np.ndarray]]): Weights of all edge models.

    Raises:
    - ValueError: If the number of layers or shapes of layers do not match between models.
    """
    for local_weights in edge_models_weights:
        if len(fog_model_weights) != len(local_weights):
            raise ValueError(f"Model shapes do not match: fog model has {len(fog_model_weights)} "
                             f"layers, but an edge model has {len(local_weights)} layers")

    for i, (fog_weight, edge_weight) in enumerate(zip(fog_model_weights, edge_models_weights[0])):
        if fog_weight.shape != edge_weight.shape:
            raise ValueError(f"Layer {i} shapes do not match: fog model layer shape is {fog_weight.shape}, "
                             f"but an edge model layer shape is {edge_weight.shape}")


def save_aggregated_model(fog_model_path, aggregated_model_weights):
    """
    Saves the aggregated model weights to the specified fog model path.

    Parameters:
    - fog_model_path (str): Path to save the aggregated model.
    - aggregated_model_weights (list[np.ndarray]): Aggregated model weights.

    Steps:
    - Loads the fog model from the given path.
    - Sets the aggregated weights to the model.
    - Compiles the model with default settings.
    - Saves the model back to the specified path.
    """
    fog_model = tf.keras.models.load_model(fog_model_path, custom_objects={'mse': MeanSquaredError()})
    fog_model.set_weights(aggregated_model_weights)
    fog_model.compile(optimizer=Adam(), loss='mse', metrics=['mse'])
    fog_model.save(fog_model_path)
    print(f"Aggregated model saved at {fog_model_path}")


def log_error(error_message):
    """
    Logs an error message and prints the traceback.

    Parameters:
    - error_message (str): The error message to log.
    """
    error = {'error': error_message}
    print(json.dumps(error), file=sys.stderr)
    traceback.print_exc()
