package com.federated_dsrl.fognode.tools.genetic.phenotype;

import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * A Data Transfer Object (DTO) representing an unevaluated genotype in a genetic algorithm.
 * <p>
 * This class is used to encapsulate a genotype structure as a list of integer alleles,
 * organized into chromosomes. It provides a simplified and serializable representation
 * of a {@link io.jenetics.Genotype}.
 * </p>
 *
 * <p>
 * Example usage:
 * <pre>
 *     // Create an unevaluated genotype
 *     UnevaluatedGenotype genotype = new UnevaluatedGenotype(List.of(
 *         List.of(1, 2, 3), // Chromosome 1
 *         List.of(4, 5, 6)  // Chromosome 2
 *     ));
 *
 *     // Access the genotype structure
 *     List<List<Integer>> alleles = genotype.getGenotype();
 *     System.out.println(alleles);
 * </pre>
 * </p>
 *
 * @see io.jenetics.Genotype
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class UnevaluatedGenotype {

    /**
     * The genotype structure represented as a list of chromosomes, where each chromosome
     * is a list of integers representing its alleles.
     */
    private List<List<Integer>> genotype;
}
