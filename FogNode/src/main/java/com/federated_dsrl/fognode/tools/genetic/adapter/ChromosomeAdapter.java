package com.federated_dsrl.fognode.tools.genetic.adapter;

import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import io.jenetics.IntegerChromosome;
import io.jenetics.Chromosome;
import io.jenetics.IntegerGene;
import io.jenetics.util.MSeq;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A custom Gson {@link TypeAdapter} for serializing and deserializing Jenetics {@link Chromosome}
 * objects with {@link IntegerGene}.
 * <p>
 * This adapter converts a chromosome to a JSON object containing an array of integer alleles
 * and reconstructs the chromosome from this JSON representation.
 * </p>
 */
public class ChromosomeAdapter extends TypeAdapter<Chromosome<IntegerGene>> {

    /**
     * Serializes a {@link Chromosome} object to its JSON representation.
     *
     * @param out        the {@link JsonWriter} to write the JSON representation
     * @param chromosome the {@link Chromosome} to serialize
     * @throws IOException if an I/O error occurs during serialization
     */
    @Override
    public void write(JsonWriter out, Chromosome<IntegerGene> chromosome) throws IOException {
        out.beginObject();
        out.name("alleles");
        out.beginArray();

        // Iterate over the genes in the chromosome
        chromosome.forEach(gene -> {
            try {
                out.value(gene.intValue());
            } catch (IOException e) {
                System.out.println("chromosome adapter error: " + e.getMessage());
            }
        });

        out.endArray();
        out.endObject();
    }

    /**
     * Deserializes a {@link Chromosome} object from its JSON representation.
     *
     * @param in the {@link JsonReader} to read the JSON representation
     * @return the deserialized {@link Chromosome} object
     * @throws IOException if an I/O error occurs during deserialization
     */
    @Override
    public Chromosome<IntegerGene> read(JsonReader in) throws IOException {
        List<Integer> alleles = new ArrayList<>();

        in.beginObject();
        while (in.hasNext()) {
            String name = in.nextName();
            if ("alleles".equals(name)) {
                in.beginArray();
                while (in.hasNext()) {
                    alleles.add(in.nextInt());
                }
                in.endArray();
            }
        }
        in.endObject();

        // Create IntegerGene instances from the alleles
        MSeq<IntegerGene> genes = MSeq.ofLength(alleles.size());
        for (int i = 0; i < alleles.size(); i++) {
            genes.set(i, IntegerGene.of(alleles.get(i), 1, 100)); // Adjust the min/max range as needed
        }

        // Convert the list of genes into an ISeq and create the IntegerChromosome
        return IntegerChromosome.of(genes.toISeq());
    }
}
