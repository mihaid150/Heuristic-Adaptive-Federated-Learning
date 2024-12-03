import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2

# Setting the environment variable to suppress TensorFlow low-level logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_initial_lstm_model(num_features=6, sequence_length=1):
    """
    Create and compile an initial LSTM model with a specified number of features and sequence length.

    Args:
    num_features (int): Number of features per timestep.
    sequence_length (int or None): Length of the input sequences. Set to None for variable length.

    Returns:
    tf.keras.Model: Compiled LSTM model.
    """
    model = tf.keras.models.Sequential([
        Input(shape=(sequence_length, num_features)),  # Define the input shape here
        LSTM(50, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        LSTM(30, activation='tanh', return_sequences=False, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(20, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def main(model_type):
    model = None

    # implemented with if conditional for future changes with another architectures
    if model_type == 'LSTM':
        model = create_initial_lstm_model(num_features=6, sequence_length=1)

    model_save_path = "/app/models/cloud"

    model.summary()

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    model_file_path = os.path.join(model_save_path, 'global_model.keras')

    model.save(model_file_path)
    print(f"Model saved to {model_file_path}")


if __name__ == "__main__":
    model_type_arg = sys.argv[1]
    main(model_type_arg)
