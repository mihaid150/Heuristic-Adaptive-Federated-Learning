package com.federated_dsrl.cloudnode.entity;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents the elapsed time metrics for fog nodes in a federated learning system.
 * <p>
 * This class tracks various time metrics related to the readiness of edges,
 * genetic evaluations, and model processing within a fog node.
 * </p>
 */
@Data
public class ElapsedTimeFog {

    /**
     * The time taken for all edges associated with the fog node to become ready, measured in seconds.
     */
    private Double timeForAllEdgesReadiness;

    /**
     * The time taken for the genetic evaluation process in the fog node, measured in seconds.
     */
    private Double timeGeneticEvaluation;

    /**
     * A list of times, measured in seconds, representing when the fog node received edge models.
     */
    private List<Double> timeReceivedEdgeModel = new ArrayList<>();

    /**
     * A list of times, measured in seconds, representing when the fog node completed aggregation of edge models.
     */
    private List<Double> timeFinishAggregation = new ArrayList<>();
}
