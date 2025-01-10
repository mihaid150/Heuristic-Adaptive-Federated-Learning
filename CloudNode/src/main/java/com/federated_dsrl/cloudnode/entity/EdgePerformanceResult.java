package com.federated_dsrl.cloudnode.entity;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * Represents the performance results of an edge device during a federated learning process.
 * <p>
 * This class captures the performance metrics for both edge and fog models on a given date.
 * </p>
 */
@Data
@AllArgsConstructor
public class EdgePerformanceResult {

    /**
     * The performance of the edge model (e.g., Mean Squared Error, Accuracy).
     */
    private Double edgeModelPerformance;

    /**
     * The performance of the fog model (e.g., Mean Squared Error, Accuracy).
     */
    private Double fogModelPerformance;

    /**
     * The date on which the performance results were recorded.
     */
    private String date;
}
