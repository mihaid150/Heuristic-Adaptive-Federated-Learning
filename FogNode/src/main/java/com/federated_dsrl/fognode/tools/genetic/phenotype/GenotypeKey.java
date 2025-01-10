package com.federated_dsrl.fognode.tools.genetic.phenotype;

import io.jenetics.Genotype;
import io.jenetics.IntegerGene;

import java.util.Objects;

/**
 * Represents a unique key for a {@link Genotype} in a genetic algorithm.
 * <p>
 * This class is used to encapsulate a {@link Genotype} as a key, providing
 * customized implementations of {@code equals()} and {@code hashCode()} for
 * efficient storage and retrieval in collections such as maps or sets.
 * </p>
 *
 * <p>
 * Example usage:
 * <pre>
 *     Genotype<IntegerGene> genotype = Genotype.of(...);
 *     GenotypeKey genotypeKey = new GenotypeKey(genotype);
 *     System.out.println(genotypeKey.equals(otherKey));
 * </pre>
 * </p>
 *
 * @param genotype The {@link Genotype} associated with this key.
 */
public record GenotypeKey(Genotype<IntegerGene> genotype) {

    /**
     * Compares this {@code GenotypeKey} to the specified object for equality.
     * <p>
     * Two {@code GenotypeKey} objects are considered equal if their {@link Genotype} values are equal.
     * </p>
     *
     * @param o The object to compare for equality.
     * @return {@code true} if the specified object is equal to this {@code GenotypeKey}; otherwise, {@code false}.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        GenotypeKey that = (GenotypeKey) o;
        return Objects.equals(genotype, that.genotype);
    }
}
