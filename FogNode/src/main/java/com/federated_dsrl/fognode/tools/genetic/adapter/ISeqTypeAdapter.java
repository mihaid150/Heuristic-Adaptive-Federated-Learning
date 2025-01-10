package com.federated_dsrl.fognode.tools.genetic.adapter;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import io.jenetics.IntegerGene;
import io.jenetics.Phenotype;
import io.jenetics.util.ISeq;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A custom Gson {@link TypeAdapter} for serializing and deserializing {@link ISeq} objects
 * containing {@link Phenotype} with {@link IntegerGene} and fitness of type {@link Double}.
 * <p>
 * This adapter allows seamless conversion of a sequence of phenotypes to and from JSON,
 * leveraging a nested {@link PhenotypeTypeAdapter} for individual phenotype serialization
 * and deserialization.
 * </p>
 */
public class ISeqTypeAdapter extends TypeAdapter<ISeq<Phenotype<IntegerGene, Double>>> {

    private final Gson gson;

    /**
     * Constructs an {@link ISeqTypeAdapter} with a custom {@link Gson} instance.
     * <p>
     * The provided {@link Gson} instance is configured to handle {@link Phenotype}
     * serialization and deserialization using {@link PhenotypeTypeAdapter}.
     * If no Gson instance is provided, a default one is created.
     * </p>
     *
     * @param gson the {@link Gson} instance for serialization and deserialization
     */
    public ISeqTypeAdapter(Gson gson) {
        this.gson = gson != null ? gson : new GsonBuilder()
                .registerTypeAdapter(Phenotype.class, new PhenotypeTypeAdapter())
                .create();
    }

    /**
     * Serializes an {@link ISeq} of {@link Phenotype} objects to its JSON representation.
     *
     * @param jsonWriter  the {@link JsonWriter} to write the JSON
     * @param phenotypes  the {@link ISeq} of phenotypes to serialize
     * @throws IOException if an I/O error occurs during serialization
     */
    @Override
    public void write(JsonWriter jsonWriter, ISeq<Phenotype<IntegerGene, Double>> phenotypes) throws IOException {
        jsonWriter.beginArray();
        for (Phenotype<IntegerGene, Double> phenotype : phenotypes) {
            gson.toJson(phenotype, Phenotype.class, jsonWriter);
        }
        jsonWriter.endArray();
    }

    /**
     * Deserializes a JSON representation into an {@link ISeq} of {@link Phenotype} objects.
     *
     * @param jsonReader the {@link JsonReader} to read the JSON
     * @return the deserialized {@link ISeq} of phenotypes
     * @throws IOException if an I/O error occurs during deserialization or if the JSON format is invalid
     */
    @Override
    public ISeq<Phenotype<IntegerGene, Double>> read(JsonReader jsonReader) throws IOException {
        List<Phenotype<IntegerGene, Double>> phenotypes = new ArrayList<>();
        jsonReader.beginArray();
        while (jsonReader.hasNext()) {
            Phenotype<IntegerGene, Double> phenotype = gson.fromJson(jsonReader, Phenotype.class);
            phenotypes.add(phenotype);
        }
        jsonReader.endArray();
        // Convert the list to ISeq
        return ISeq.of(phenotypes);
    }
}
