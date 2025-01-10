import sys
from model_utils import init_new_model, load_model_weights, validate_model_shapes, save_aggregated_model, log_error


def aggregate_models(global_model_path, model_paths, is_first_aggregation):
    """
    Aggregate model weights by averaging weights from multiple fog models and the global model.

    Args:
        global_model_path (str): Path to the global model file containing weights.
        model_paths (list): List of paths to fog model weight files to be aggregated.
        is_first_aggregation (bool): Flag indicating whether this is the first aggregation.
                                     If True, only fog model weights are averaged.
                                     If False, the global model is included in the averaging.

    Returns:
        list: Aggregated weights for the model.

    Raises:
        Exception: If any error occurs during aggregation, it will be logged.
    """
    try:
        # Load global model weights and layers
        global_model_weights, global_model_layers = load_model_weights(global_model_path)

        # Load fog model weights
        fog_models_data = [load_model_weights(path) for path in model_paths]
        fog_models_weights = [data[0] for data in fog_models_data]

        # Validate that all models have the same shape
        validate_model_shapes(global_model_weights, fog_models_weights)

        # Initialize a new model with zero weights
        aggregated_model = init_new_model(global_model_weights)

        # Aggregate weights
        if is_first_aggregation:
            # Average only the fog model weights
            for i in range(len(global_model_weights)):
                for fog_weight in fog_models_weights:
                    aggregated_model[i] += fog_weight[i]
                aggregated_model[i] /= len(fog_models_weights)
        else:
            # Average between fog model weights and the global model weights
            for i in range(len(global_model_weights)):
                aggregated_model[i] = global_model_weights[i]  # Start with the global model weights
                for fog_weight in fog_models_weights:
                    aggregated_model[i] += fog_weight[i]
                aggregated_model[i] /= (len(fog_models_weights) + 1)  # Average with the global model

        return aggregated_model

    except Exception as e:
        log_error(str(e))


if __name__ == '__main__':
    """
    Main script for aggregating model weights.

    Command-line Arguments:
        1. global_model_path_main_scope: Path to the global model file.
        2. is_first_aggregation_arg: Boolean ('true' or 'false') indicating if this is the first aggregation.
        3. fog_model_paths: Paths to the fog model files to be aggregated.

    Example Usage:
        python script.py global_model.h5 true fog_model1.h5 fog_model2.h5 fog_model3.h5
    """
    # Read command-line arguments
    global_model_path_main_scope = sys.argv[1]
    is_first_aggregation_arg = sys.argv[2].lower() == 'true'
    fog_model_paths = sys.argv[3:]

    # Perform aggregation
    aggregated_model_weights = aggregate_models(global_model_path_main_scope, fog_model_paths, is_first_aggregation_arg)

    # Save aggregated model weights
    save_aggregated_model(global_model_path_main_scope, aggregated_model_weights)
