package com.federated_dsrl.fognode.tools.genetic.adapter;

import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import com.federated_dsrl.fognode.tools.genetic.phenotype.GenotypeKey;
import io.jenetics.Chromosome;
import io.jenetics.Genotype;
import io.jenetics.IntegerChromosome;
import io.jenetics.IntegerGene;
import io.jenetics.util.ISeq;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A custom Gson {@link TypeAdapter} for serializing and deserializing {@link GenotypeKey} objects.
 * <p>
 * This adapter converts a {@link GenotypeKey} to a JSON representation and reconstructs it
 * from the JSON structure. The underlying {@link Genotype} contains {@link IntegerGene}.
 * </p>
 */
public class GenotypeKeyAdapter extends TypeAdapter<GenotypeKey> {

    /**
     * Serializes a {@link GenotypeKey} to its JSON representation.
     *
     * @param jsonWriter   the {@link JsonWriter} to write the JSON
     * @param genotypeKey  the {@link GenotypeKey} to serialize
     * @throws IOException if an I/O error occurs during serialization
     */
    @Override
    public void write(JsonWriter jsonWriter, GenotypeKey genotypeKey) throws IOException {
        jsonWriter.beginObject();
        jsonWriter.name("genotype");

        // Start writing an array to hold the chromosomes
        jsonWriter.beginArray();
        for (Chromosome<IntegerGene> chromosome : genotypeKey.genotype()) {
            jsonWriter.beginArray();
            for (IntegerGene gene : chromosome) {
                jsonWriter.value(gene.intValue()); // Write each gene value
            }
            jsonWriter.endArray();
        }
        jsonWriter.endArray();

        jsonWriter.endObject();
    }

    /**
     * Deserializes a JSON representation into a {@link GenotypeKey}.
     *
     * @param jsonReader the {@link JsonReader} to read the JSON
     * @return the deserialized {@link GenotypeKey}
     * @throws IOException if an I/O error occurs during deserialization or if the JSON format is invalid
     */
    @Override
    public GenotypeKey read(JsonReader jsonReader) throws IOException {
        jsonReader.beginObject();
        String name = jsonReader.nextName();
        if (!"genotype".equals(name)) {
            throw new IOException("Expected 'genotype' but was " + name);
        }

        List<Chromosome<IntegerGene>> chromosomes = new ArrayList<>();
        jsonReader.beginArray();
        while (jsonReader.hasNext()) {
            List<IntegerGene> genes = new ArrayList<>();
            jsonReader.beginArray();
            while (jsonReader.hasNext()) {
                int alleleValue = jsonReader.nextInt();
                genes.add(IntegerGene.of(alleleValue, 1, 100)); // Assuming a default range of [1, 100]
            }
            jsonReader.endArray();
            chromosomes.add(IntegerChromosome.of(ISeq.of(genes)));
        }
        jsonReader.endArray();

        jsonReader.endObject();

        Genotype<IntegerGene> genotype = Genotype.of(chromosomes);
        return new GenotypeKey(genotype);
    }
}
