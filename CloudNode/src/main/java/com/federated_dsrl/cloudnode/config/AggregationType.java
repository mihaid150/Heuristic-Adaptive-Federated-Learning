package com.federated_dsrl.cloudnode.config;

/**
 * Represents the types of aggregation methods available in the federated learning framework developed.
 * <p>
 * There are two aggregation methods:
 * </p>
 * <ul>
 *     <li><b>GENETIC:</b> Where the child Fog node utilizes the Genetic Engine Component along with
 *     additional tools for enhanced training. This method leverages genetic algorithms to optimize model
 *     aggregation and training outcomes.</li>
 *     <li><b>AVERAGE:</b> Employs the classical FedAvg method, which calculates the average of the models
 *     contributed by participating nodes.</li>
 * </ul>
 */
public enum AggregationType {
    /**
     * Genetic aggregation method, using genetic algorithms for optimization.
     */
    GENETIC,
    /**
     * Average aggregation method, employing the traditional FedAvg technique.
     */
    AVERAGE
}
