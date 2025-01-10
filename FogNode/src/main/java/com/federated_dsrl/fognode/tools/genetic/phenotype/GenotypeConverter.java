package com.federated_dsrl.fognode.tools.genetic.phenotype;

import io.jenetics.Chromosome;
import io.jenetics.Genotype;
import io.jenetics.IntegerGene;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for converting between Genotype and GenotypeDTO.
 */
public class GenotypeConverter {

    /**
     * Converts a Jenetics Genotype<IntegerGene> to a GenotypeDTO.
     *
     * @param genotype the genotype to convert
     * @return the corresponding GenotypeDTO
     */
    public static UnevaluatedGenotype toDTO(Genotype<IntegerGene> genotype) {
        List<List<Integer>> genotypeList = new ArrayList<>();
        for (Chromosome<IntegerGene> chromosome : genotype) {
            List<Integer> chromosomeList = new ArrayList<>();
            for (IntegerGene gene : chromosome) {
                chromosomeList.add(gene.intValue());
            }
            genotypeList.add(chromosomeList);
        }
        return new UnevaluatedGenotype(genotypeList);
    }
}
