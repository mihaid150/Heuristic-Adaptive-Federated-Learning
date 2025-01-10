package com.federated_dsrl.cloudnode.config;

/**
 * Enumeration representing different strategies for evaluating and updating
 * individuals in a genetic algorithm.
 * <p>These strategies determine how the population of individuals is updated
 * during the genetic evolution process.</p>
 */
public enum GeneticEvaluationStrategy {
    /**
     * Updates all individuals in the population.
     * <p>Every individual in the population is evaluated and updated during
     * each iteration of the genetic algorithm.</p>
     */
    UPDATE_ALL_INDIVIDUALS,

    /**
     * Updates only the bottom-performing individuals in the population.
     * <p>Focuses on improving the least-performing individuals, leaving the
     * top-performing ones unchanged.</p>
     */
    UPDATE_BOTTOM_INDIVIDUALS,

    /**
     * Updates a random subset of individuals in the population.
     * <p>Randomly selects individuals for evaluation and update, which can
     * introduce more diversity into the population.</p>
     */
    UPDATE_RANDOM
}
