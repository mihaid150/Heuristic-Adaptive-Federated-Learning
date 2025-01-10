import sys
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model_utils import init_new_model, load_model_weights, validate_model_shapes, save_aggregated_model, log_error


def scale_lambda_prevs(lambda_prevs):
    """
    Scale `lambda_prevs` values for weighted aggregation.

    If all values are identical, they are uniformly set to 0.5. Otherwise,
    they are scaled between 0.1 and 0.9 to avoid exact values at the edges.

    Args:
        lambda_prevs (list): List of lambda coefficients for aggregation.

    Returns:
        numpy.ndarray: Scaled `lambda_prevs` for weighted aggregation.
    """
    if np.all(lambda_prevs == lambda_prevs[0]):
        return np.full_like(lambda_prevs, 0.5)

    # Reshape for MinMaxScaler
    lambda_prevs = np.array(lambda_prevs).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0.1, 0.9))  # Avoid exact 0 and 1
    scaled = scaler.fit_transform(lambda_prevs).flatten()

    logging.info('Scaled lambda_prevs: ' + str(scaled))
    return scaled


def aggregate_models(global_model_path, model_paths, lambda_prevs):
    """
    Perform weighted aggregation of model weights using specified coefficients.

    Args:
        global_model_path (str): Path to the global model file.
        model_paths (list): List of paths to fog model weight files.
        lambda_prevs (list): Coefficients for weighted aggregation.

    Returns:
        list: Aggregated weights for the global model.

    Raises:
        Exception: Logs and raises any error that occurs during aggregation.
    """
    try:
        for index, lambda_prev in enumerate(lambda_prevs):
            logging.info(f"lambda {index} is {lambda_prev}")

        # Load global and fog models
        global_model_weights, global_model_layers = load_model_weights(global_model_path)
        fog_models_data = [load_model_weights(path) for path in model_paths]
        fog_models_weights = [data[0] for data in fog_models_data]

        # Validate that all models have the same shape
        validate_model_shapes(global_model_weights, fog_models_weights)

        # Initialize a new model with zero weights
        aggregated_model = init_new_model(global_model_weights)
        weight_sum = sum(lambda_prevs)  # Sum of coefficients for weighted average

        # Perform weighted aggregation
        for i in range(len(global_model_weights)):
            for j, local_weight in enumerate(lambda_prevs):
                aggregated_model[i] += local_weight * fog_models_weights[j][i]
            aggregated_model[i] /= weight_sum

        return aggregated_model
    except Exception as e:
        log_error(str(e))


if __name__ == '__main__':
    """
    Main script for weighted aggregation of models.

    Command-line Arguments:
        1. global_model_path_main_scope (str): Path to the global model file.
        2. fog_model_paths (list of str): Paths to the fog model files, alternating with weights.
        3. lambda_prevs_main_scope (list of float): Coefficients for weighted aggregation.

    Example Usage:
        python script.py global_model.h5 fog_model1.h5 0.4 fog_model2.h5 0.6

    Process:
        1. Scale the lambda coefficients for normalized weighted aggregation.
        2. Aggregate the global and fog model weights.
        3. Save the aggregated model to the specified global model path.
    """
    # Parse command-line arguments
    global_model_path_main_scope = sys.argv[1]
    fog_model_paths = sys.argv[2::2]  # Extract alternate arguments as model paths
    lambda_prevs_main_scope = [float(w) for w in sys.argv[3::2]]  # Extract alternate arguments as weights

    # Log initial lambda coefficients
    logging.info('lambda_prevs_main_scope: ' + str(lambda_prevs_main_scope))

    # Scale the lambda coefficients
    scaled_lambda_prevs = list(scale_lambda_prevs(lambda_prevs_main_scope))
    logging.info('Scaled lambda_prevs_main_scope: ' + str(scaled_lambda_prevs))

    # Perform model aggregation
    aggregated_model_weights = aggregate_models(global_model_path_main_scope, fog_model_paths, scaled_lambda_prevs)

    # Save the aggregated model
    save_aggregated_model(global_model_path_main_scope, aggregated_model_weights)
