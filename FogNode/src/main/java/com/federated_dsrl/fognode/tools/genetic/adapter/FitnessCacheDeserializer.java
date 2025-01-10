package com.federated_dsrl.fognode.tools.genetic.adapter;

import com.federated_dsrl.fognode.tools.genetic.phenotype.GenotypeKey;
import com.google.gson.*;
import io.jenetics.Chromosome;
import io.jenetics.Genotype;
import io.jenetics.IntegerChromosome;
import io.jenetics.IntegerGene;
import io.jenetics.util.ISeq;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Custom deserializer for a map of {@link GenotypeKey} to {@link Double}, representing the fitness cache.
 * <p>
 * This deserializer uses Gson to parse a JSON representation of genotypes and their fitness values
 * into a Java {@code Map<GenotypeKey, Double>} object. Each genotype is reconstructed as a
 * {@link Genotype} of {@link IntegerGene}.
 * </p>
 */
public class FitnessCacheDeserializer implements JsonDeserializer<Map<GenotypeKey, Double>> {

    /**
     * Deserializes a JSON element into a map of {@link GenotypeKey} to {@link Double}.
     *
     * @param json    the JSON element to deserialize
     * @param typeOfT the type of the object to deserialize
     * @param context the deserialization context
     * @return a map where keys are {@link GenotypeKey} and values are their corresponding fitness scores
     * @throws JsonParseException if the JSON structure is invalid or cannot be parsed
     */
    @Override
    public Map<GenotypeKey, Double> deserialize(JsonElement json, Type typeOfT, JsonDeserializationContext context) throws JsonParseException {
        Map<GenotypeKey, Double> map = new HashMap<>();

        // Access the array inside the "genotypes" object
        JsonObject jsonObject = json.getAsJsonObject();
        JsonArray genotypesArray = jsonObject.getAsJsonArray("genotypes");

        for (JsonElement element : genotypesArray) {
            JsonObject genotypeObject = element.getAsJsonObject().getAsJsonObject("genotype");
            JsonArray genotypeArray = genotypeObject.getAsJsonArray("genotype");

            // Deserialize the genotype array back into a Genotype<IntegerGene>
            List<Chromosome<IntegerGene>> chromosomes = new ArrayList<>();
            for (JsonElement chromosomeElement : genotypeArray) {
                JsonArray chromosomeArray = chromosomeElement.getAsJsonArray();
                List<IntegerGene> genes = new ArrayList<>();
                for (JsonElement geneElement : chromosomeArray) {
                    int allele = geneElement.getAsInt();
                    genes.add(IntegerGene.of(allele, Integer.MIN_VALUE, Integer.MAX_VALUE));
                }
                chromosomes.add(IntegerChromosome.of(ISeq.of(genes)));
            }
            Genotype<IntegerGene> genotype = Genotype.of(chromosomes);

            // Create the GenotypeKey from the deserialized Genotype
            GenotypeKey key = new GenotypeKey(genotype);
            Double value = element.getAsJsonObject().get("fitness").getAsDouble();

            // Put the key-value pair into the map
            map.put(key, value);
        }

        return map;
    }
}
