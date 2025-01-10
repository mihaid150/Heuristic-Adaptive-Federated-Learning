import sys
from model_utils import init_new_model, load_model_weights, validate_model_shapes, save_aggregated_model, log_error


def aggregate_models(fog_model_path, model_paths, is_first_aggregation):
    """
    Aggregates model weights from multiple edge models and the fog model.

    Parameters:
    - fog_model_path (str): Path to the fog model file.
    - model_paths (list[str]): List of paths to edge model files.
    - is_first_aggregation (bool): Flag indicating whether this is the first aggregation.

    Returns:
    - list[numpy.ndarray]: Aggregated model weights.

    Steps:
    - Loads the fog model weights.
    - Loads edge model weights from the provided paths.
    - Validates the shapes of all models to ensure compatibility.
    - Initializes a new model with zero weights.
    - Aggregates weights based on whether it's the first aggregation or not:
        - First aggregation averages only edge models.
        - Subsequent aggregations average fog and edge models.
    """
    try:
        # Load weights and layer structures of the fog model
        fog_model_weights, fog_model_layers = load_model_weights(fog_model_path)
        # Load weights of all edge models
        edge_models_data = [load_model_weights(path) for path in model_paths]
        edge_models_weights = [data[0] for data in edge_models_data]

        # Validate shapes of all models
        validate_model_shapes(fog_model_weights, edge_models_weights)

        # Initialize a new model with zero weights
        aggregated_model = init_new_model(fog_model_weights)

        # Aggregate weights
        if is_first_aggregation:
            # Average weights of all edge models
            for i in range(len(fog_model_weights)):
                for edge_weight in edge_models_weights:
                    aggregated_model[i] += edge_weight[i]
                aggregated_model[i] /= len(edge_models_weights)
        else:
            # Average weights of the fog model and all edge models
            for i in range(len(fog_model_weights)):
                aggregated_model[i] = fog_model_weights[i]  # Start with fog model weights
                for edge_weight in edge_models_weights:
                    aggregated_model[i] += edge_weight[i]
                aggregated_model[i] /= (len(edge_models_weights) + 1)  # Include the fog model in the average

        return aggregated_model

    except Exception as e:
        log_error(str(e))


if __name__ == '__main__':
    """
    Main function for aggregating models.

    Usage:
    python aggregate_models.py <fog_model_path> <is_first_aggregation> <edge_model_path1> <edge_model_path2> ...

    Arguments:
    - fog_model_path (str): Path to the fog model file.
    - is_first_aggregation (str): "true" or "false" indicating if this is the first aggregation.
    - edge_model_paths (list[str]): Paths to edge model files.

    Steps:
    - Parses command-line arguments.
    - Calls `aggregate_models` with the provided arguments.
    - Saves the aggregated model weights to the fog model path.
    """
    fog_model_path_main_scope = sys.argv[1]
    is_first_aggregation_arg = sys.argv[2].lower() == 'true'
    edge_model_paths = sys.argv[3:]

    aggregated_model_weights = aggregate_models(fog_model_path_main_scope, edge_model_paths, is_first_aggregation_arg)
    save_aggregated_model(fog_model_path_main_scope, aggregated_model_weights)
