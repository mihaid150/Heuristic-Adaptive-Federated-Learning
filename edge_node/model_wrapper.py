import tensorflow as tf


class MoELSTMWithGateLoss(tf.keras.Model):
    def __init__(self, base_model, alpha=0.1):
        super().__init__()
        self.base_model = base_model
        self.alpha = alpha
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self._gate_model = tf.keras.Model(
            inputs=self.base_model.input,
            outputs=self.base_model.get_layer("gate").output,
        )

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def compile(self, optimizer):
        super().compile(optimizer=optimizer)
        self.loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.mse_tracker = tf.keras.metrics.Mean(name="mse_loss")
        self.gate_tracker = tf.keras.metrics.Mean(name="gate_loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.mse_tracker, self.gate_tracker]

    def train_step(self, data):
        X, y_true = data  # y_true must be a tuple (regression_target, spike_flag)
        reg_target, spike_flag = y_true

        with tf.GradientTape() as tape:
            output = self.base_model(X, training=True)
            gate_output = self._gate_model(X, training=True)

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

    def test_step(self, data):
        X, y_true = data
        reg_target, spike_flag = y_true

        output = self.base_model(X, training=False)
        gate_output = self._gate_model(X, training=False)
        mse_loss = self.loss_fn(reg_target, output)
        gate_loss = self.bce(spike_flag, gate_output)
        total_loss = mse_loss + self.alpha * gate_loss

        self.loss_tracker.update_state(total_loss)
        self.mse_tracker.update_state(mse_loss)
        self.gate_tracker.update_state(gate_loss)

        return {
            "loss": self.loss_tracker.result(),
            "mse_loss": self.mse_tracker.result(),
            "gate_loss": self.gate_tracker.result(),
        }