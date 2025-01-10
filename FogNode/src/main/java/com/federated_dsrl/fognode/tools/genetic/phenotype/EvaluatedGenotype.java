package com.federated_dsrl.fognode.tools.genetic.phenotype;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a genotype that has been evaluated, including its associated fitness value.
 * <p>
 * This class encapsulates a {@link UnevaluatedGenotype} and its corresponding fitness score,
 * allowing for easy storage and manipulation of evaluated genetic algorithm results.
 * </p>
 *
 * <p>
 * Example usage:
 * <pre>
 *     UnevaluatedGenotype unevaluatedGenotype = new UnevaluatedGenotype(...);
 *     Double fitness = 42.0;
 *     EvaluatedGenotype evaluatedGenotype = new EvaluatedGenotype(unevaluatedGenotype, fitness);
 * </pre>
 * </p>
 *
 * <p>
 * An evaluated genotype consists of:
 * <ul>
 *     <li>{@link UnevaluatedGenotype}: The genotype structure without its fitness value.</li>
 *     <li>{@link Double}: The fitness value calculated for the genotype.</li>
 * </ul>
 * </p>
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class EvaluatedGenotype {

    /**
     * The unevaluated genotype containing the genetic structure.
     */
    private UnevaluatedGenotype genotype;

    /**
     * The fitness value associated with the genotype.
     */
    private Double fitness;
}
