import sys
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model_utils import init_new_model, load_model_weights, validate_model_shapes, save_aggregated_model, log_error


def scale_lambda_prevs(lambda_prevs):
    """ Scale lambda_prevs such that if all are equal, they are set to 0.5.
    Otherwise, scale between 0 and 1 but not exactly at the edges.
    """
    if np.all(lambda_prevs == lambda_prevs[0]):
        return np.full_like(lambda_prevs, 0.5)

    # reshape for MinMaxScaler
    lambda_prevs = np.array(lambda_prevs).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0.1, 0.9))  # avoid exact 0 and 1
    scaled = scaler.fit_transform(lambda_prevs).flatten()

    logging.info('Scaled lambda_prevs: ' + str(scaled))
    return scaled


def aggregate_models(global_model_path, model_paths, lambda_prevs):
    try:
        for index, lambda_prev in enumerate(lambda_prevs):
            logging.info(f"lambda {index} is {lambda_prev}")

        global_model_weights, global_model_layers = load_model_weights(global_model_path)
        fog_models_data = [load_model_weights(path) for path in model_paths]
        fog_models_weights = [data[0] for data in fog_models_data]

        # validate that all models have the same shape
        validate_model_shapes(global_model_weights, fog_models_weights)

        # initialize a new model with zero weights
        aggregated_model = init_new_model(global_model_weights)
        weight_sum = sum(lambda_prevs)  # sum of coefficients for weighted average

        # perform weighted aggregation for standard layers
        for i in range(len(global_model_weights)):
            for j, local_weight in enumerate(lambda_prevs):
                aggregated_model[i] += local_weight * fog_models_weights[j][i]
            aggregated_model[i] /= weight_sum

        return aggregated_model
    except Exception as e:
        log_error(str(e))


if __name__ == '__main__':
    global_model_path_main_scope = sys.argv[1]
    fog_model_paths = sys.argv[2::2]
    lambda_prevs_main_scope = [float(w) for w in sys.argv[3::2]]
    logging.info('lambda_prevs_main_scope: ' + str(lambda_prevs_main_scope))

    # scale the lambda_prevs for aggregation
    scaled_lambda_prevs = scale_lambda_prevs(lambda_prevs_main_scope)
    logging.info('Scaled lambda_prevs_main_scope: ' + str(scaled_lambda_prevs))

    # perform the model aggregation
    aggregated_model_weights = aggregate_models(global_model_path_main_scope, fog_model_paths, scaled_lambda_prevs)

    # save the aggregated model
    save_aggregated_model(global_model_path_main_scope, aggregated_model_weights)
