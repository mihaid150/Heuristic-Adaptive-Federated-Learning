package org.federated_dsrl.edgenode.config;

import lombok.Data;
import org.springframework.stereotype.Component;

/**
 * Manages training parameters for model training, such as learning rate, batch size,
 * number of epochs, early stopping patience, and the number of fine-tune layers.
 */
@Component
@Data
public class TrainingParametersManager {

    /**
     * The learning rate for training the model.
     */
    private Double learningRate;

    /**
     * The batch size used during training.
     */
    private Integer batchSize;

    /**
     * The number of training epochs.
     */
    private Integer epochs;

    /**
     * The early stopping patience to prevent overfitting.
     */
    private Integer earlyStoppingPatience;

    /**
     * The number of fine-tune layers to adjust during transfer learning.
     */
    private Integer fineTuneLayers;

    /**
     * The type of model being used for training. Default is "LSTM".
     */
    private String modelType = "LSTM";

    /**
     * Provides a string representation of the training parameters for debugging or logging.
     *
     * @return a string containing the current training parameters.
     */
    @Override
    public String toString() {
        return "TrainingParametersManager{" +
                "learningRate=" + learningRate +
                ", batchSize=" + batchSize +
                ", epochs=" + epochs +
                ", earlyStoppingPatience=" + earlyStoppingPatience +
                ", fineTuneLayers=" + fineTuneLayers +
                ", modelType='" + modelType + '\'' +
                '}';
    }
}
