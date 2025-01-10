package com.federated_dsrl.fognode.tools.genetic.adapter;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import io.jenetics.Genotype;
import io.jenetics.IntegerGene;
import io.jenetics.Phenotype;

import java.io.IOException;

/**
 * A custom Gson {@link TypeAdapter} for serializing and deserializing {@link Phenotype} objects
 * with {@link IntegerGene} and fitness of type {@link Double}.
 * <p>
 * This adapter converts a {@link Phenotype} to a JSON representation containing its genotype,
 * generation, and fitness, and reconstructs it from this JSON format.
 * </p>
 */
public class PhenotypeTypeAdapter extends TypeAdapter<Phenotype<IntegerGene, Double>> {
    private final Gson gson;

    /**
     * Constructs a {@link PhenotypeTypeAdapter} with a custom {@link Gson} instance.
     * <p>
     * The {@link Gson} instance is configured with a {@link GenotypeTypeAdapter} to handle
     * serialization and deserialization of {@link Genotype} objects.
     * </p>
     */
    public PhenotypeTypeAdapter() {
        this.gson = new GsonBuilder()
                .registerTypeAdapter(Genotype.class, new GenotypeTypeAdapter())
                .create();
    }

    /**
     * Serializes a {@link Phenotype} to its JSON representation.
     *
     * @param jsonWriter the {@link JsonWriter} to write the JSON
     * @param phenotype  the {@link Phenotype} to serialize
     * @throws IOException if an I/O error occurs during serialization
     */
    @Override
    public void write(JsonWriter jsonWriter, Phenotype<IntegerGene, Double> phenotype) throws IOException {
        jsonWriter.beginObject();
        jsonWriter.name("genotype");
        gson.toJson(phenotype.genotype(), Genotype.class, jsonWriter);
        jsonWriter.name("generation");
        jsonWriter.value(phenotype.generation());

        jsonWriter.name("fitness");
        if (phenotype.fitnessOptional().isPresent()) {
            jsonWriter.value(phenotype.fitness());
        } else {
            jsonWriter.nullValue();
        }

        jsonWriter.endObject();
    }

    /**
     * Deserializes a JSON representation into a {@link Phenotype}.
     *
     * @param jsonReader the {@link JsonReader} to read the JSON
     * @return the deserialized {@link Phenotype}
     * @throws IOException if an I/O error occurs during deserialization or if required fields are missing
     */
    @Override
    public Phenotype<IntegerGene, Double> read(JsonReader jsonReader) throws IOException {
        jsonReader.beginObject();
        Genotype<IntegerGene> genotype = null;
        long generation = 1;
        Double fitness = null;

        while (jsonReader.hasNext()) {
            String name = jsonReader.nextName();
            switch (name) {
                case "genotype":
                    genotype = gson.fromJson(jsonReader, Genotype.class);
                    break;
                case "generation":
                    generation = jsonReader.nextLong();
                    break;
                case "fitness":
                    if (jsonReader.peek() != com.google.gson.stream.JsonToken.NULL) {
                        fitness = jsonReader.nextDouble();
                    } else {
                        jsonReader.nextNull();
                    }
                    break;
                default:
                    jsonReader.skipValue();
                    break;
            }
        }
        jsonReader.endObject();

        if (genotype == null) {
            throw new IOException("Genotype is required to deserialize a Phenotype");
        }

        if (fitness != null) {
            return Phenotype.of(genotype, generation, fitness);
        } else {
            return Phenotype.of(genotype, generation);
        }
    }
}
