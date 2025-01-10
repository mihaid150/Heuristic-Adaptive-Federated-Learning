package com.federated_dsrl.fognode.tools.genetic.phenotype;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for saving and loading fitness cache.
 */
public class FitnessCacheManager {

    private static final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * Saves the fitness cache to a JSON file.
     *
     * @param evaluatedPhenotypes the list of evaluated phenotypes to save
     * @param filePath            the path to the JSON file
     * @throws IOException if an I/O error occurs
     */
    public static void saveFitnessCache(List<EvaluatedPhenotype> evaluatedPhenotypes, String filePath) throws IOException {
        List<EvaluatedGenotype> dtoList = new ArrayList<>();
        for (EvaluatedPhenotype evaluatedPhenotype : evaluatedPhenotypes) {
            UnevaluatedGenotype unevaluatedGenotype = GenotypeConverter.toDTO(evaluatedPhenotype.getPhenotype().genotype());
            EvaluatedGenotype evaluatedGenotype = new EvaluatedGenotype(unevaluatedGenotype, evaluatedPhenotype.getLastFitness());
            dtoList.add(evaluatedGenotype);
        }
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(new File(filePath), dtoList);
    }
}
