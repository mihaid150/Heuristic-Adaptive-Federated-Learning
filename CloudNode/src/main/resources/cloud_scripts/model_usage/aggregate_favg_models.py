import sys
from model_utils import init_new_model, load_model_weights, validate_model_shapes, save_aggregated_model, log_error


def aggregate_models(global_model_path, model_paths, is_first_aggregation):
    try:
        global_model_weights, global_model_layers = load_model_weights(global_model_path)
        fog_models_data = [load_model_weights(path) for path in model_paths]

        fog_models_weights = [data[0] for data in fog_models_data]

        # validate that all models have the same shape
        validate_model_shapes(global_model_weights, fog_models_weights)

        # initialize a new model with zero weights
        aggregated_model = init_new_model(global_model_weights)

        # aggregate models
        if is_first_aggregation:
            # average only the fog model weights
            for i in range(len(global_model_weights)):
                for fog_weight in fog_models_weights:
                    aggregated_model[i] += fog_weight[i]
                aggregated_model[i] /= len(fog_models_weights)
        else:
            # average between fog model weights and the global model weights
            for i in range(len(global_model_weights)):
                aggregated_model[i] = global_model_weights[i]  # start with the global model weights
                for fog_weight in fog_models_weights:
                    aggregated_model[i] += fog_weight[i]
                aggregated_model[i] /= (len(fog_models_weights) + 1)  # average with the global model

        return aggregated_model

    except Exception as e:
        log_error(str(e))


if __name__ == '__main__':
    global_model_path_main_scope = sys.argv[1]
    is_first_aggregation_arg = sys.argv[2].lower() == 'true'
    fog_model_paths = sys.argv[3:]

    aggregated_model_weights = aggregate_models(global_model_path_main_scope, fog_model_paths, is_first_aggregation_arg)
    save_aggregated_model(global_model_path_main_scope, aggregated_model_weights)
