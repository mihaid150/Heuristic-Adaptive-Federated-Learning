package org.federated_dsrl.edgenode.entity;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * Represents the performance results of edge and fog models for a specific date.
 */
@Data
@AllArgsConstructor
public class PerformanceResult {

    /**
     * The performance metric of the edge model (e.g., mse loss ).
     */
    private Double edgeModelPerformance;

    /**
     * The performance metric of the fog model (e.g., mse loss).
     */
    private Double fogModelPerformance;

    /**
     * The date associated with the performance results.
     */
    private String date;
}
