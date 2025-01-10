import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler
import joblib
from data_usage.preprocessor import preprocess_training_data


def add_synthetic_rows(df, target_length):
    """
    Adds synthetic rows to the DataFrame until it reaches the target length by duplicating
    the existing rows with added small random noise.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_length (int): Target number of rows for the DataFrame.

    Returns:
        pd.DataFrame: DataFrame extended with synthetic rows.
    """
    while len(df) < target_length:
        synthetic_data = df.sample(frac=1, replace=True, random_state=42).copy()
        for col in synthetic_data.select_dtypes(include=[np.number]).columns:
            synthetic_data[col] += np.random.normal(loc=0, scale=0.01, size=synthetic_data[col].shape)
        df = pd.concat([df, synthetic_data], ignore_index=True)
        if len(df) > target_length:
            df = df.iloc[:target_length]
    return df


def prepare_features(df, sequence_length=48):
    """
    Prepares sliding windows of features and targets for time series prediction.

    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
        sequence_length (int): Number of timesteps in each input sequence.

    Returns:
        tuple: A tuple of (features, targets) where:
               - features is an array of shape (num_samples, sequence_length, num_features)
               - targets is an array of shape (num_samples,)
    """
    if len(df) < 96:
        print("WARNING: Dataset has fewer than 96 rows. Adding synthetic rows...")
        df = add_synthetic_rows(df, target_length=96)

    def generate_sliding_windows(data, sequence_length):
        features, targets = [], []
        for i in range(len(data) - sequence_length):
            features.append(data[i:i + sequence_length, :-1])
            targets.append(data[i + sequence_length, -1])
        return np.array(features), np.array(targets)

    raw_data = df[['value', 'value_rolling_mean_3', 'value_rolling_max_3', 'value_rolling_min_3',
                   'value_rolling_mean_6', 'value_rolling_max_6', 'value']].values
    features, targets = generate_sliding_windows(raw_data, sequence_length)

    print(f"Generated features shape: {features.shape}")
    print(f"Generated target shape: {targets.shape}")

    if features.size == 0 or targets.size == 0:
        raise ValueError("Features or targets have no values after processing.")

    return features, targets


def evaluate_model(data_path, model_file, date):
    """
    Evaluates a trained LSTM model on a given dataset and generates day-ahead predictions.

    Args:
        data_path (str): Path to the JSON file containing the dataset.
        model_file (str): Path to the trained model file (.keras).
        date (str): Date of the evaluation, used for saving results.

    Saves:
        A CSV file containing the real and predicted values, as well as metrics (R2, MAE, MSE, RMSE).
    """
    try:
        preprocess_training_data(data_path)
        df = pd.read_json(data_path)

        features, target = prepare_features(df, sequence_length=48)

        feature_scaler_path = '/app/cache_json/feature_scaler.pkl'
        target_scaler_path = '/app/cache_json/target_scaler.pkl'

        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)

        features_flattened = features.reshape(-1, features.shape[-1])
        features_scaled = feature_scaler.transform(features_flattened).reshape(features.shape)

        custom_objects = {'mse': MeanSquaredError()}
        model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)

        next_day_predictions = predict_day_ahead(features_scaled[-1], model, target_scaler)

        real_values_next_day = df['value'].iloc[-48:].values

        r2 = r2_score(real_values_next_day, next_day_predictions)
        mse = mean_squared_error(real_values_next_day, next_day_predictions)
        mae = np.mean(np.abs(real_values_next_day - next_day_predictions))
        rmse = np.sqrt(mse)

        predictions_df = pd.DataFrame({
            'date': [date] * 48,
            'real_values': real_values_next_day,
            'predicted_values': next_day_predictions,
            'r2': [r2] * 48,
            'mae': [mae] * 48,
            'mse': [mse] * 48,
            'rmse': [rmse] * 48
        })

        output_path = "/app/evaluation/predictions_day_ahead.csv"
        if os.path.exists(output_path):
            predictions_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            predictions_df.to_csv(output_path, index=False)

        print(f"Day-ahead predictions and metrics saved to {output_path}")

    except Exception as e:
        error = {'error in evaluate_model': str(e)}
        print(json.dumps(error), file=sys.stderr)


def predict_day_ahead(features, model, scaler, sequence_length=48):
    """
    Generates day-ahead predictions recursively using a trained model.

    Args:
        features (np.ndarray): Input sequence of shape (sequence_length, num_features).
        model (tf.keras.Model): Trained model to predict the next value.
        scaler (StandardScaler): Scaler for inverse-transforming predictions.
        sequence_length (int): Number of timesteps in the input sequence.

    Returns:
        list: List of 48 predicted values for the next day.
    """
    predictions = []
    input_sequence = features[-sequence_length:]

    for _ in range(sequence_length):
        input_sequence_reshaped = input_sequence.reshape(1, sequence_length, -1)
        predicted_scaled = model.predict(input_sequence_reshaped, verbose=0).squeeze()
        predicted_value = scaler.inverse_transform([[predicted_scaled]]).flatten()[0]
        predictions.append(predicted_value)
        next_input = np.append(input_sequence[1:], np.array([[predicted_value] + [0] * (input_sequence.shape[1] - 1)]), axis=0)
        input_sequence = next_input

    return predictions


if __name__ == "__main__":
    """
    Main script entry point. Accepts data_path, model_file, and date as command-line arguments.
    """
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Not all arguments are provided. Expected 3 arguments: data_path, model_file, date'}))
    else:
        data_path_param = sys.argv[1]
        model_file_param = sys.argv[2]
        date_param = sys.argv[3]
        evaluate_model(data_path_param, model_file_param, date_param)
