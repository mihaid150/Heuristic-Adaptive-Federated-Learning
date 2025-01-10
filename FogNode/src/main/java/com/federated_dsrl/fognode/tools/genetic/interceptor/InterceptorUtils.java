package com.federated_dsrl.fognode.tools.genetic.interceptor;

import com.federated_dsrl.fognode.tools.genetic.engine.GeneticEngine;
import com.federated_dsrl.fognode.tools.genetic.phenotype.EvaluatedPhenotype;
import io.jenetics.*;
import io.jenetics.util.BaseSeq;
import io.jenetics.util.ISeq;
import lombok.Setter;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Random;

/**
 * Utility class for shared genetic algorithm functionality used in interceptors.
 *
 * @param <G> the gene type
 * @param <C> the fitness result type, which must be comparable
 */
@Component
public class InterceptorUtils<G extends Gene<?, G>, C extends Comparable<? super C>> {

    private final GeneticEngine geneticEngine;
    private final Mutator<G, C> mutator;
    private final SinglePointCrossover<G, C> crossover;
    @Setter
    private List<Phenotype<G, C>> population;

    /**
     * Constructs an instance of {@code InterceptorUtils}.
     *
     * @param geneticEngine the genetic engine used for fitness evaluation
     */
    public InterceptorUtils(GeneticEngine geneticEngine) {
        this.geneticEngine = geneticEngine;
        this.mutator = new Mutator<>(0.1); // 10% mutation probability
        this.crossover = new SinglePointCrossover<>(0.6); // 60% crossover probability
    }

    /**
     * Safely applies mutation and crossover to a phenotype.
     *
     * @param phenotype  the phenotype to alter
     * @return the altered phenotype
     */
    public Phenotype<G, C> safeApplyMutationCrossover(Phenotype<G, C> phenotype) {
        try {
            return applyMutationCrossover(phenotype);
        } catch (Exception e) {
            System.out.println("Error during mutation and crossover for phenotype: " + phenotype + ". Error: " + e.getMessage());
            return phenotype; // Return the original phenotype in case of error
        }
    }

    /**
     * Applies mutation and crossover to a phenotype.
     *
     * @param phenotype  the phenotype to alter
     * @return the altered phenotype
     */
    public Phenotype<G, C> applyMutationCrossover(Phenotype<G, C> phenotype) {
        // Select a random mate from the population
        Phenotype<G, C> matePhenotype = population.get(new Random().nextInt(population.size()));

        // Create a sequence of phenotypes for crossover
        ISeq<Phenotype<G, C>> phenotypeISeq = ISeq.of(phenotype, matePhenotype);

        // Apply crossover
        ISeq<Phenotype<G, C>> crossedPhenotypes = crossover.alter(phenotypeISeq, 1).population();

        // Select the resulting phenotype after crossover
        Phenotype<G, C> crossedPhenotype = crossedPhenotypes.get(0);

        // Apply mutation
        ISeq<Phenotype<G, C>> mutatedPhenotypes = mutator.alter(ISeq.of(crossedPhenotype), 1).population();

        // Return the mutated phenotype
        return mutatedPhenotypes.get(0);
    }

    /**
     * Converts a list of phenotypes to an {@link ISeq}.
     *
     * @param phenotypes the list of phenotypes
     * @return the converted {@code ISeq}
     */
    public ISeq<Phenotype<G, C>> toISeq(List<Phenotype<G, C>> phenotypes) {
        return ISeq.of(phenotypes);
    }

    /**
     * Updates the list of evaluated phenotypes with the new population.
     *
     * @param newPopulation  the new population of phenotypes
     * @param allIndividuals the list of evaluated phenotypes to update
     */
    public void updateAllIndividuals(ISeq<Phenotype<G, C>> newPopulation, List<EvaluatedPhenotype> allIndividuals) {
        for (Phenotype<G, C> phenotype : newPopulation) {
            int[] values = phenotype.genotype().stream()
                    .flatMap(BaseSeq::stream)
                    .mapToInt(gene -> ((IntegerGene) gene).intValue())
                    .toArray();

            Double fitness = geneticEngine.evaluate(values);

            boolean found = false;
            for (EvaluatedPhenotype evalPhenotype : allIndividuals) {
                if (evalPhenotype.getPhenotype().genotype().equals(phenotype.genotype())) {
                    evalPhenotype.setLastFitness(fitness);
                    found = true;
                    break;
                }
            }

            if (!found) {
                // Add new phenotype to allIndividuals
                @SuppressWarnings("unchecked")
                Phenotype<IntegerGene, Double> castedPhenotype = (Phenotype<IntegerGene, Double>) phenotype;
                allIndividuals.add(new EvaluatedPhenotype(castedPhenotype, fitness));
            }
        }
    }
}
