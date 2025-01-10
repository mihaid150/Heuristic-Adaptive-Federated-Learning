package com.federated_dsrl.fognode.tools.genetic.interceptor;

/**
 * Enumeration representing the strategies for updating the genetic population during evolution.
 * <p>
 * These strategies define how individuals in the population are selected for evaluation and update:
 * <ul>
 *     <li>{@code UPDATE_ALL_INDIVIDUALS} - Evaluate and update all individuals in the population.</li>
 *     <li>{@code UPDATE_BOTTOM_INDIVIDUALS} - Retain the top-performing individuals with the best fitness
 *     and update the bottom-performing individuals.</li>
 *     <li>{@code UPDATE_RANDOM} - Update a random subset of individuals in the population.</li>
 * </ul>
 */
public enum GeneticEvaluationStrategy {
    /**
     * Evaluate and update all individuals in the population.
     */
    UPDATE_ALL_INDIVIDUALS,

    /**
     * Retain the top-performing individuals with the best fitness
     * and update the bottom-performing individuals.
     */
    UPDATE_BOTTOM_INDIVIDUALS,

    /**
     * Update a random subset of individuals in the population.
     */
    UPDATE_RANDOM
}
