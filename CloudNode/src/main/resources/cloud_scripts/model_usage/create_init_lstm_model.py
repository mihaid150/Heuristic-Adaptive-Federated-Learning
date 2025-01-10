import os
import sys
import tensorflow as tf

# Setting the environment variable to suppress TensorFlow low-level logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_initial_lstm_model():
    """
    Create and compile an initial LSTM model with a specified number of features and sequence length.

    Returns:
    tf.keras.Model: Compiled LSTM model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(48, 6), dtype=tf.float32),
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def main(model_type):
    """
    Main function to create and save the initial LSTM model based on the specified model type.

    Args:
    model_type (str): The type of model to create. Currently supports 'LSTM'.
    """
    model = None

    # Implemented with if conditional for future changes with other architectures
    if model_type == 'LSTM':
        model = create_initial_lstm_model()

    model_save_path = "/app/models/cloud"

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    model_file_path = os.path.join(model_save_path, 'global_model.keras')

    model.save(model_file_path)
    print(f"Model saved to {model_file_path}")


if __name__ == "__main__":
    model_type_arg = sys.argv[1]
    main(model_type_arg)
