import tensorflow as tf
from shared.utils import required_columns, CombineExperts, select_gate_inputs
from shared.logging_config import logger


def create_initial_lstm_model(sequence_length=144, mask_value=-1):
    """
    Create and compile an enhanced LSTM model for time series forecasting.

    The model includes:
      - A 1D convolution layer to capture local temporal patterns.
      - Batch normalization and dropout for regularization.
      - Two LSTM layers with tanh activations.
      - Dense layers to learn non-linear relationships.

    Parameters:
        sequence_length (int): The length of the input sequences.
        mask_value (int): The value to use for masking the input sequences.

    Returns:
        tf.keras.Model: A compiled LSTM model.
    """
    num_features = len(required_columns) - 1
    inputs = tf.keras.layers.Input(shape=(sequence_length, num_features), dtype=tf.float32)

    # Process inputs with Conv1D first (without masking)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Apply Masking after the convolution block.
    x = tf.keras.layers.Masking(mask_value=mask_value)(x)

    # LSTM layers for sequential modeling
    x = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)(x)
    x = tf.keras.layers.LSTM(128, activation='tanh')(x)

    # Dense layers for non-linear transformations
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='mse', metrics=["mae", "mse"])

    logger.info(f"Created model with input shape ({sequence_length}, {num_features})")
    return model


def create_attention_lstm_model(sequence_length=144, mask_value=-1):
    """
    Create and compile an advanced LSTM model for time series forecasting,
    incorporating multi-scale convolutions, bidirectional LSTMs, and an attention mechanism.

    Parameters:
        sequence_length (int): The length of the input sequences.
        mask_value (int): The value to use for masking the input sequences.

    Returns:
        tf.keras.Model: A compiled model.
    """
    num_features = len(required_columns) - 1
    inputs = tf.keras.layers.Input(shape=(sequence_length, num_features), dtype=tf.float32)

    # Multi-scale convolution block: use different kernel sizes
    conv3 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    conv5 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(inputs)
    conv7 = tf.keras.layers.Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Concatenate()([conv3, conv5, conv7])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Apply Masking to ignore padded timesteps (if zeros are used for padding)
    x = tf.keras.layers.Masking(mask_value=mask_value)(x)

    # Bidirectional LSTM layers with return_sequences=True for attention
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True, dropout=0.2)
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True, dropout=0.2)
    )(x)

    # Self-attention layer using MultiHeadAttention
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    # Use a residual connection to combine the attention output with the original LSTM output.
    x = tf.keras.layers.Add()([x, attention_output])
    # Global average pooling over time dimension.
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dense layers for final prediction.
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=["mae", "mse"])

    logger.info(f"Created advanced model with input shape ({sequence_length}, {num_features})")
    return model


def create_enhanced_attention_lstm_model(sequence_length=144, mask_value=-1):
    """
    Create and compile an enhanced LSTM model for time series forecasting,
    incorporating multi-scale convolutions and a self-attention mechanism while
    preserving the core LSTM layers.

    The model includes:
      - Multi-scale Conv1D layers to capture local patterns.
      - Batch normalization and dropout for regularization.
      - A Masking layer applied after the convolution block.
      - A first LSTM layer (return_sequences=True) followed by a MultiHeadAttention block.
      - A residual connection combining the attention output with the first LSTM output.
      - A second LSTM layer (return_sequences=False) to produce a summary vector.
      - Dense layers for the final regression output.

    Parameters:
        sequence_length (int): The length of the input sequences.
        mask_value (int): The value to use for masking the input sequences.

    Returns:
        tf.keras.Model: A compiled LSTM model.
    """
    num_features = len(required_columns) - 1
    inputs = tf.keras.layers.Input(shape=(sequence_length, num_features), dtype=tf.float32)

    # Multi-scale convolution block: two parallel Conv1D layers with different kernel sizes.
    conv3 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    conv5 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Concatenate()([conv3, conv5])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Apply Masking layer to ignore padded zeros.
    x = tf.keras.layers.Masking(mask_value=mask_value)(x)

    # First LSTM layer with return_sequences=True.
    lstm_out1 = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)(x)

    # Self-attention mechanism on the outputs of the first LSTM.
    # Using MultiHeadAttention to allow the model to reweight timesteps.
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(lstm_out1, lstm_out1)
    # Residual connection: add the attention output to the original LSTM output.
    x_att = tf.keras.layers.Add()([lstm_out1, attention_output])

    # Second LSTM layer (core preserved) to summarize the sequence.
    lstm_out2 = tf.keras.layers.LSTM(128, activation='tanh')(x_att)

    # Dense layers for non-linear transformation and final prediction.
    x_dense = tf.keras.layers.Dense(64, activation='relu')(lstm_out2)
    x_dense = tf.keras.layers.Dropout(0.3)(x_dense)
    outputs = tf.keras.layers.Dense(1)(x_dense)

    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='mse', metrics=["mae", "mse"])

    logger.info(f"Created enhanced LSTM model with input shape ({sequence_length}, {num_features})")
    return model

def create_moe_lstm_model(sequence_length=144, mask_value=-1):
    """Create a mixture-of-experts model with a gating network."""
    num_features = len(required_columns) - 1
    inputs = tf.keras.layers.Input(shape=(sequence_length, num_features), dtype=tf.float32)

    # Shared backbone

    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Masking(mask_value=mask_value)(x)
    x = tf.keras.layers.LSTM(64, activation='tanh')(x)
    shared = tf.keras.layers.Dense(32, activation='relu')(x)

    # Expert 1: normal regime
    normal = tf.keras.layers.Dense(16, activation="relu")(shared)
    normal = tf.keras.layers.Dense(
        1, activation="linear", name="normal_head"
    )(normal)

    # Expert 2: spike regime
    spike = tf.keras.layers.Dense(16, activation="relu")(shared)
    spike = tf.keras.layers.Dense(
        1, activation="linear", name="spike_head"
    )(spike)

    # Gating network outputs value in [0,1]
    gate_in = tf.keras.layers.Lambda(select_gate_inputs, name="gate_selector")(inputs)
    gate_x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(gate_in)
    gate_x = tf.keras.layers.GlobalMaxPooling1D()(gate_x)
    gate_x = tf.keras.layers.Dense(64, activation="relu")(gate_x)
    gate_x = tf.keras.layers.Dropout(0.3)(gate_x)
    gate = tf.keras.layers.Dense(1, activation="sigmoid", name="gate")(gate_x)

    # Combine experts using the gating value without anonymous lambda
    outputs = CombineExperts()([normal, spike, gate])

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=["mae", "mse"])

    logger.info(
        f"Created mixture-of-experts model with input shape ({sequence_length}, {num_features})"
    )
    return model