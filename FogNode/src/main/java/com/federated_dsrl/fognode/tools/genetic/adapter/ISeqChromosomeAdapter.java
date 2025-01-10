package com.federated_dsrl.fognode.tools.genetic.adapter;

import com.google.gson.TypeAdapter;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import com.google.gson.stream.JsonWriter;
import io.jenetics.util.ISeq;
import io.jenetics.Chromosome;
import io.jenetics.IntegerGene;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A custom Gson {@link TypeAdapter} for serializing and deserializing {@link ISeq} objects
 * containing {@link Chromosome} with {@link IntegerGene}.
 * <p>
 * This adapter uses a nested {@link ChromosomeAdapter} to handle individual chromosome serialization
 * and deserialization and supports both array and object representations for chromosomes in JSON.
 * </p>
 */
public class ISeqChromosomeAdapter extends TypeAdapter<ISeq<Chromosome<IntegerGene>>> {

    private final Gson gson;

    /**
     * Constructs an {@link ISeqChromosomeAdapter} with a custom {@link Gson} instance.
     * The {@link Gson} instance is configured with a {@link ChromosomeAdapter} to handle
     * {@link Chromosome} serialization and deserialization.
     */
    public ISeqChromosomeAdapter() {
        this.gson = new GsonBuilder()
                .registerTypeAdapter(Chromosome.class, new ChromosomeAdapter())
                .create();
    }

    /**
     * Serializes an {@link ISeq} of {@link Chromosome} objects to its JSON representation.
     *
     * @param out   the {@link JsonWriter} to write the JSON
     * @param value the {@link ISeq} of chromosomes to serialize
     * @throws IOException if an I/O error occurs during serialization
     */
    @Override
    public void write(JsonWriter out, ISeq<Chromosome<IntegerGene>> value) throws IOException {
        out.beginArray();
        for (Chromosome<IntegerGene> chromosome : value) {
            gson.toJson(chromosome, Chromosome.class, out);
        }
        out.endArray();
    }

    /**
     * Deserializes a JSON representation into an {@link ISeq} of {@link Chromosome} objects.
     * <p>
     * The method supports JSON arrays of chromosomes or a single chromosome object.
     * </p>
     *
     * @param in the {@link JsonReader} to read the JSON
     * @return the deserialized {@link ISeq} of chromosomes
     * @throws IOException if an I/O error occurs during deserialization or if the JSON format is invalid
     */
    @Override
    public ISeq<Chromosome<IntegerGene>> read(JsonReader in) throws IOException {
        List<Chromosome<IntegerGene>> chromosomes = new ArrayList<>();
        if (in.peek() == JsonToken.BEGIN_ARRAY) {
            in.beginArray(); // Read as array
            while (in.hasNext()) {
                Chromosome<IntegerGene> chromosome = gson.fromJson(in, Chromosome.class);
                chromosomes.add(chromosome);
            }
            in.endArray(); // End array
        } else if (in.peek() == JsonToken.BEGIN_OBJECT) {
            Chromosome<IntegerGene> chromosome = gson.fromJson(in, Chromosome.class);
            chromosomes.add(chromosome);
        } else {
            throw new IllegalStateException("Expected BEGIN_ARRAY or BEGIN_OBJECT but was " + in.peek());
        }
        return ISeq.of(chromosomes);
    }
}
