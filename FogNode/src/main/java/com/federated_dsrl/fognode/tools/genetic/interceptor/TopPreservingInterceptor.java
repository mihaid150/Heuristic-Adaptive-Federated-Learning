package com.federated_dsrl.fognode.tools.genetic.interceptor;

import com.federated_dsrl.fognode.tools.genetic.phenotype.EvaluatedPhenotype;
import com.federated_dsrl.fognode.tools.genetic.engine.GeneticEngine;
import io.jenetics.*;
import io.jenetics.engine.EvolutionInterceptor;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.util.ISeq;

import java.util.Comparator;
import java.util.List;

/**
 * A custom {@link EvolutionInterceptor} implementation that preserves a specified number of top-performing
 * individuals in a genetic algorithm population and applies mutation and crossover to the remaining individuals.
 *
 * @param <G> the gene type
 * @param <C> the fitness result type, which must be comparable
 */
public class TopPreservingInterceptor<G extends Gene<?, G>, C extends Comparable<? super C>>
        implements EvolutionInterceptor<G, C> {

    private final int topPreserveCount;
    private final List<EvaluatedPhenotype> allIndividuals;
    private final InterceptorUtils<G, C> interceptorUtils;

    /**
     * Constructs a new instance of {@code TopPreservingInterceptor}.
     *
     * @param topPreserveCount the number of top-performing individuals to preserve
     * @param allIndividuals   the list of all evaluated phenotypes
     * @param geneticEngine    the genetic engine for fitness evaluation
     */
    public TopPreservingInterceptor(int topPreserveCount, List<EvaluatedPhenotype> allIndividuals,
                                    GeneticEngine geneticEngine) {
        this.topPreserveCount = topPreserveCount;
        this.allIndividuals = allIndividuals;
        this.interceptorUtils = new InterceptorUtils<>(geneticEngine);
    }

    @Override
    public EvolutionResult<G, C> after(EvolutionResult<G, C> result) {
        System.out.println("TopPreservingInterceptor: Starting after() method.");

        // Sort the population by fitness
        ISeq<Phenotype<G, C>> sortedPopulation = result.population()
                .stream()
                .sorted(Comparator.comparing(Phenotype::fitness))
                .collect(ISeq.toISeq());

        System.out.println("TopPreservingInterceptor: Population sorted by fitness.");

        // Preserve the top individuals
        ISeq<Phenotype<G, C>> preservedTop = sortedPopulation.subSeq(0, topPreserveCount);
        ISeq<Phenotype<G, C>> remainingPopulation = sortedPopulation.subSeq(topPreserveCount, sortedPopulation.size());

        System.out.println("TopPreservingInterceptor: Preserved top " + topPreserveCount + " individuals.");

        // Set the population in the utils
        interceptorUtils.setPopulation(result.population().asList());

        // Apply mutation and crossover to the remaining population
        ISeq<Phenotype<G, C>> alteredPopulation = remainingPopulation.stream()
                .map(interceptorUtils::safeApplyMutationCrossover)
                .collect(ISeq.toISeq());

        // Combine preserved top and altered population
        ISeq<Phenotype<G, C>> newPopulation = preservedTop.append(alteredPopulation);

        // Update the population in allIndividuals
        interceptorUtils.updateAllIndividuals(newPopulation, allIndividuals);

        return EvolutionResult.of(result.optimize(), newPopulation, result.generation(), result.durations(),
                result.killCount(), result.invalidCount(), result.alterCount());
    }
}

