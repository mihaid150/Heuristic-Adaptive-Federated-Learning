package com.federated_dsrl.fognode.tools.genetic.interceptor;

import com.federated_dsrl.fognode.tools.genetic.phenotype.EvaluatedPhenotype;
import com.federated_dsrl.fognode.tools.genetic.phenotype.GenotypeKey;
import io.jenetics.IntegerGene;
import io.jenetics.Phenotype;
import io.jenetics.engine.EvolutionInterceptor;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.util.BaseSeq;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Custom {@link EvolutionInterceptor} that updates all individuals in the genetic population by applying
 * mutation and crossover. The interceptor also maintains a fitness cache to avoid redundant evaluations.
 */
public class UpdateAllIndividualsInterceptor implements EvolutionInterceptor<IntegerGene, Double> {

    private final List<EvaluatedPhenotype> allIndividuals;
    private final Map<GenotypeKey, Double> fitnessCache;
    private final Function<int[], Double> evaluate;
    private final String date;
    private final InterceptorUtils<IntegerGene, Double> interceptorUtils;

    /**
     * Constructs an instance of {@code UpdateAllIndividualsInterceptor}.
     *
     * @param allIndividuals  the list of all evaluated phenotypes
     * @param fitnessCache    the fitness cache to avoid redundant evaluations
     * @param evaluate        the function for evaluating fitness
     * @param date            the current training date
     * @param interceptorUtils the utility class for mutation and crossover operations
     */
    public UpdateAllIndividualsInterceptor(
            List<EvaluatedPhenotype> allIndividuals,
            Map<GenotypeKey, Double> fitnessCache,
            Function<int[], Double> evaluate,
            String date,
            InterceptorUtils<IntegerGene, Double> interceptorUtils) {
        this.allIndividuals = allIndividuals;
        this.fitnessCache = fitnessCache;
        this.evaluate = evaluate;
        this.date = date;
        this.interceptorUtils = interceptorUtils;
    }

    /**
     * Updates the population by applying mutation and crossover, and evaluates the fitness for all individuals.
     *
     * @param result the result of the evolution step
     * @return the updated evolution result
     */
    @Override
    public EvolutionResult<IntegerGene, Double> after(EvolutionResult<IntegerGene, Double> result) {
        // Set the current population in the InterceptorUtils
        List<Phenotype<IntegerGene, Double>> population = new ArrayList<>(result.population().asList());
        interceptorUtils.setPopulation(population);

        // Clear and prepare the altered population
        allIndividuals.clear();
        List<Phenotype<IntegerGene, Double>> alteredPopulation = new ArrayList<>();

        for (Phenotype<IntegerGene, Double> phenotype : population) {
            // Apply mutation and crossover safely
            Phenotype<IntegerGene, Double> alteredPhenotype = interceptorUtils.safeApplyMutationCrossover(phenotype);
            alteredPopulation.add(alteredPhenotype);

            // Extract parameter values from the altered phenotype
            int[] values = alteredPhenotype.genotype().stream()
                    .flatMap(BaseSeq::stream)
                    .mapToInt(IntegerGene::intValue)
                    .toArray();

            // Evaluate fitness and update caches
            Double fitness = evaluate.apply(values);
            allIndividuals.add(new EvaluatedPhenotype(alteredPhenotype, fitness));
            fitnessCache.put(new GenotypeKey(alteredPhenotype.genotype()), fitness);
        }

        // Log the updated population
        System.out.println("Updated Population for date " + date + ":");
        for (EvaluatedPhenotype evalPhenotype : allIndividuals) {
            System.out.println("Genotype: " + evalPhenotype.getPhenotype().genotype() +
                    ", Fitness: " + evalPhenotype.getLastFitness());
        }

        // Return the updated evolution result
        return EvolutionResult.of(
                result.optimize(),
                interceptorUtils.toISeq(alteredPopulation),
                result.generation(),
                result.durations(),
                result.killCount(),
                result.invalidCount(),
                result.alterCount()
        );
    }
}
