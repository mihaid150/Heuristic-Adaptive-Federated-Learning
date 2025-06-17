import tensorflow as tf


class MoELSTMWithGateLoss(tf.keras.Model):
    def __init__(self, base_model, alpha=0.1):
        super().__init__()
        self.base_model = base_model
        self.alpha = alpha
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.bce = tf.keras.losses.BinaryCrossentropy()

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.mse_tracker = tf.keras.metrics.Mean(name="mse_loss")
        self.gate_tracker = tf.keras.metrics.Mean(name="gate_loss")

    def train_step(self, data):
        X, y_true = data  # y_true must be a tuple (regression_target, spike_flag)
        reg_target, spike_flag = y_true

        print("reg_target shape:", tf.shape(reg_target))
        print("spike_flag shape:", tf.shape(spike_flag))
        print("spike_flag dtype:", spike_flag.dtype)
        print("Has None?", spike_flag is None)

        with tf.GradientTape() as tape:
            output = self.base_model(X, training=True)
            gate_output = self.base_model.get_layer("gate").output

            # Compute losses
            mse_loss = self.loss_fn(reg_target, output)
            gate_loss = self.bce(spike_flag, gate_output)
            total_loss = mse_loss + self.alpha * gate_loss

        grads = tape.gradient(total_loss, self.base_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_weights))

        # Track metrics
        self.loss_tracker.update_state(total_loss)
        self.mse_tracker.update_state(mse_loss)
        self.gate_tracker.update_state(gate_loss)

        return {
            "loss": self.loss_tracker.result(),
            "mse_loss": self.mse_tracker.result(),
            "gate_loss": self.gate_tracker.result()
        }
