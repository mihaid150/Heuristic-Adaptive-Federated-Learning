package com.federated_dsrl.fognode.tools.genetic.adapter;

import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import io.jenetics.Genotype;
import io.jenetics.IntegerChromosome;
import io.jenetics.IntegerGene;
import io.jenetics.util.ISeq;
import io.jenetics.util.MSeq;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A custom Gson {@link TypeAdapter} for serializing and deserializing {@link Genotype} objects
 * containing {@link IntegerGene}.
 * <p>
 * This adapter converts a {@link Genotype} to a JSON representation and reconstructs it
 * from the JSON structure, preserving properties like min, max, length, and allele values
 * for each chromosome.
 * </p>
 */
public class GenotypeTypeAdapter extends TypeAdapter<Genotype<IntegerGene>> {

    /**
     * Serializes a {@link Genotype} to its JSON representation.
     *
     * @param out      the {@link JsonWriter} to write the JSON
     * @param genotype the {@link Genotype} to serialize
     * @throws IOException if an I/O error occurs during serialization
     */
    @Override
    public void write(JsonWriter out, Genotype<IntegerGene> genotype) throws IOException {
        out.beginObject();
        out.name("chromosomes");
        out.beginArray();

        genotype.stream().forEach(chromosome -> {
            try {
                if (chromosome instanceof IntegerChromosome intChromosome) {
                    out.beginObject();

                    int min = intChromosome.get(0).min();
                    int max = intChromosome.get(0).max();
                    out.name("min").value(min);
                    out.name("max").value(max);
                    out.name("length").value(intChromosome.length());

                    out.name("alleles");
                    out.beginArray();
                    intChromosome.forEach(gene -> {
                        try {
                            out.value(gene.intValue());
                        } catch (IOException e) {
                            System.out.println("genotype type adapter int chromosome error: " + e.getMessage());
                        }
                    });
                    out.endArray();
                    out.endObject();
                    System.out.println("Serialized Chromosome: " + intChromosome);
                }
            } catch (IOException e) {
                System.out.println("genotype type adapter error: " + e.getMessage());
            }
        });

        out.endArray();
        out.endObject();
    }

    /**
     * Deserializes a JSON representation into a {@link Genotype}.
     *
     * @param in the {@link JsonReader} to read the JSON
     * @return the deserialized {@link Genotype}
     * @throws IOException if an I/O error occurs during deserialization or if the JSON format is invalid
     */
    @Override
    public Genotype<IntegerGene> read(JsonReader in) throws IOException {
        List<IntegerChromosome> chromosomes = new ArrayList<>();

        in.beginObject();
        while (in.hasNext()) {
            String name = in.nextName();
            if ("chromosomes".equals(name)) {
                in.beginArray();

                while (in.hasNext()) {
                    in.beginObject();

                    Integer min = null;
                    Integer max = null;
                    List<Integer> alleles = new ArrayList<>();

                    while (in.hasNext()) {
                        String propName = in.nextName();
                        switch (propName) {
                            case "min":
                                min = in.nextInt();
                                break;
                            case "max":
                                max = in.nextInt();
                                break;
                            case "alleles":
                                in.beginArray();
                                while (in.hasNext()) {
                                    alleles.add(in.nextInt());
                                }
                                in.endArray();
                                break;
                            default:
                                in.skipValue();
                                break;
                        }
                    }

                    in.endObject();

                    if (min == null || max == null || alleles.isEmpty()) {
                        throw new IOException("Incomplete chromosome data in JSON.");
                    }

                    MSeq<IntegerGene> genes = MSeq.ofLength(alleles.size());
                    for (int i = 0; i < alleles.size(); i++) {
                        genes.set(i, IntegerGene.of(alleles.get(i), min, max));
                    }

                    IntegerChromosome chromosome = IntegerChromosome.of(genes.toISeq());
                    chromosomes.add(chromosome);
                    System.out.println("Deserialized Chromosome: " + chromosome);
                }

                in.endArray();
            } else {
                in.skipValue();
            }
        }
        in.endObject();

        return Genotype.of(ISeq.of(chromosomes));
    }
}
