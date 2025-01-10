import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
import os
import sys


def create_enhanced_lstm_model():
    """
    Create and compile an enhanced LSTM model for time-series or sequential data.

    The model consists of:
    - Two LSTM layers with ReLU activation and L2 regularization.
    - Dropout layers to prevent overfitting.
    - A Dense layer with ReLU activation for feature extraction.
    - A Dense output layer for regression tasks.

    Input shape is set to (48, 6), suitable for 48 timesteps and 6 features per timestep.

    Returns:
    --------
    tf.keras.Model
        Compiled LSTM model with the Adam optimizer and mean squared error loss.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(48, 6), dtype=tf.float32),
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=True, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        tf.keras.layers.LSTM(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def main(model_type):
    """
    Main function to create and save the specified model type.

    If `model_type` is 'LSTM', an enhanced LSTM model is created and saved to the file system.

    Args:
    -----
    model_type : str
        The type of model to create. Currently only supports 'LSTM'.

    Raises:
    -------
    ValueError
        If the `model_type` is unsupported.

    Notes:
    ------
    The model is saved to the '/app/models/' directory as 'enhanced_model.keras'.
    """
    if model_type == 'LSTM':
        model = create_enhanced_lstm_model()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_save_path = "/app/models/"

    # Print model summary
    model.summary()

    # Ensure the save directory exists
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    model_file_path = os.path.join(model_save_path, 'enhanced_model.keras')

    # Save the model to the specified path
    model.save(model_file_path)
    print(f"Model saved to {model_file_path}")


if __name__ == "__main__":
    """
    Entry point of the script.

    Accepts the model type as a command-line argument and saves the corresponding model.
    """
    if len(sys.argv) != 2:
        print("Usage: python <script_name.py> <model_type>")
        sys.exit(1)

    model_type_arg = sys.argv[1]
    main(model_type_arg)
