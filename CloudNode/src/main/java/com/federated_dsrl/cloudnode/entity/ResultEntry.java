package com.federated_dsrl.cloudnode.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents an entry in the result dataset for a specific edge or device.
 * <p>
 * Each entry contains information about the performance of both the original and retrained models
 * for a given logical device or edge.
 * </p>
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class ResultEntry {

    /**
     * The Logical Communication Layer Identifier (LCLID) of the edge or device.
     */
    private String lclid;

    /**
     * The performance metric (e.g., accuracy, mean squared error) of the original model before retraining.
     */
    private double originalModelPerformance;

    /**
     * The performance metric (e.g., accuracy, mean squared error) of the retrained model after federated updates.
     */
    private double retrainedModelPerformance;
}
