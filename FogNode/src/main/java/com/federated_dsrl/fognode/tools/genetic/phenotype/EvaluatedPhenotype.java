package com.federated_dsrl.fognode.tools.genetic.phenotype;

import io.jenetics.IntegerGene;
import io.jenetics.Phenotype;
import lombok.Getter;
import lombok.Setter;

/**
 * Represents an evaluated phenotype in a genetic algorithm.
 * <p>
 * This class encapsulates a {@link Phenotype} and its associated fitness value,
 * allowing for tracking and updating the fitness of the phenotype over time.
 * </p>
 *
 * <p>
 * Example usage:
 * <pre>
 *     Phenotype<IntegerGene, Double> phenotype = Phenotype.of(...);
 *     Double fitness = 42.0;
 *     EvaluatedPhenotype evaluatedPhenotype = new EvaluatedPhenotype(phenotype, fitness);
 *     System.out.println(evaluatedPhenotype);
 * </pre>
 * </p>
 *
 * <p>
 * An evaluated phenotype consists of:
 * <ul>
 *     <li>{@link Phenotype}: The phenotype representing a solution in the genetic algorithm.</li>
 *     <li>{@link Double}: The last known fitness value for the phenotype, which can be updated as needed.</li>
 * </ul>
 * </p>
 */
@Getter
public class EvaluatedPhenotype {

    /**
     * The {@link Phenotype} representing the genetic algorithm solution.
     */
    private final Phenotype<IntegerGene, Double> phenotype;

    /**
     * The last known fitness value of the phenotype.
     * Defaults to {@link Double#MAX_VALUE} if not evaluated.
     */
    @Setter
    private Double lastFitness = Double.MAX_VALUE;

    /**
     * Constructs a new {@code EvaluatedPhenotype} with the given phenotype and fitness value.
     *
     * @param phenotype The {@link Phenotype} to be evaluated.
     * @param fitness   The initial fitness value of the phenotype.
     */
    public EvaluatedPhenotype(Phenotype<IntegerGene, Double> phenotype, Double fitness) {
        this.phenotype = phenotype;
        this.lastFitness = fitness;
    }

    /**
     * Returns a string representation of the evaluated phenotype, including its genotype and fitness value.
     *
     * @return A string representation of the evaluated phenotype.
     */
    @Override
    public String toString() {
        return "Phenotype: " + phenotype.genotype() + ", Fitness: " +
                (lastFitness != null ? lastFitness : "Not evaluated");
    }
}
