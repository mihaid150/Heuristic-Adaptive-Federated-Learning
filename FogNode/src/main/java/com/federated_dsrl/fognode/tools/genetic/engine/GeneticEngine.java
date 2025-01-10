package com.federated_dsrl.fognode.tools.genetic.engine;

import com.federated_dsrl.fognode.config.DeviceManager;
import com.federated_dsrl.fognode.entity.EdgeEntity;
import com.federated_dsrl.fognode.tools.ConcurrencyManager;
import com.federated_dsrl.fognode.tools.genetic.adapter.*;
import com.federated_dsrl.fognode.tools.genetic.interceptor.GeneticEvaluationStrategy;
import com.federated_dsrl.fognode.tools.genetic.interceptor.InterceptorUtils;
import com.federated_dsrl.fognode.tools.genetic.interceptor.TopPreservingInterceptor;
import com.federated_dsrl.fognode.tools.genetic.interceptor.UpdateAllIndividualsInterceptor;
import com.federated_dsrl.fognode.tools.genetic.phenotype.*;
import com.federated_dsrl.fognode.tools.simulated_annealing.CloudCoolingSchedule;
import com.federated_dsrl.fognode.utils.ModelFileHandler;
import com.google.gson.*;
import io.jenetics.*;
import io.jenetics.engine.Codecs;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.engine.InvertibleCodec;
import io.jenetics.util.BaseSeq;
import io.jenetics.util.ISeq;
import io.jenetics.util.IntRange;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import com.google.gson.reflect.TypeToken;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.*;

/**
 * The {@code GeneticEngine} class manages the genetic algorithm for hyperparameter optimization
 * in a federated fog node system. This class handles:
 * <ul>
 *     <li>Population initialization</li>
 *     <li>Fitness evaluation</li>
 *     <li>Genetic evolution</li>
 *     <li>State persistence and restoration</li>
 * </ul>
 */
@Component
@RequiredArgsConstructor
@Data
public class GeneticEngine {
    private List<String> date;
    private final ModelFileHandler modelFileHandler;
    private final DeviceManager deviceManager;
    private final ConcurrencyManager concurrencyManager;
    private final List<EvaluatedPhenotype> allIndividuals = new ArrayList<>();
    private EdgeEntity evaluationEdge;
    private final CloudCoolingSchedule cloudCoolingSchedule;
    private final Double BOLTZMANN_CONSTANT = 1.380649e-23;
    private final Double ADDITIONAL_FACTOR = 1.e23;
    private final Map<GenotypeKey, Double> fitnessCache = Collections.synchronizedMap(
            new LinkedHashMap<>() {
                @Override
                protected boolean removeEldestEntry(Map.Entry<GenotypeKey, Double> eldest) {
                    return this.size() > 100;
                }
            }
    );
    private final String INDIVIDUALS_FILE = "cache_json/individuals_cache.json";
    private final String FITNESS_CACHE_FILE = "cache_json/fitness_cache.json";
    private double bestFitness = Double.MAX_VALUE;
    private int stagnationCounter = 0;
    private static final int MAX_STAGNATION = 1;
    private static final int TOP_PRESERVE_COUNT = 2;
    private final int POPULATION_SIZE = 5;
    private final int NUMBER_GENERATIONS = 4;

    /**
     * Initializes the population with random individuals for hyperparameter optimization.
     * <p>
     * Each individual is a {@link Phenotype} representing a unique set of hyperparameters:
     * <ul>
     *     <li>Learning rate</li>
     *     <li>Batch size</li>
     *     <li>Number of epochs</li>
     *     <li>Early stopping patience</li>
     *     <li>Number of fine-tuning layers</li>
     * </ul>
     * </p>
     */
    public void initializePopulation() {
        allIndividuals.clear();
        fitnessCache.clear();

        Random random = new Random();
        System.out.println("Initializing population...");

        for (int i = 0; i < POPULATION_SIZE; i++) {
            // Generate random values for each parameter within the specified ranges
            int learningRate = random.nextInt(100) + 1;
            int batchSize = random.nextInt(113) + 16;
            int numberEpochs = random.nextInt(100) + 1;
            int earlyStoppingPatience = random.nextInt(20) + 1;
            int numberOfFineTuningLayers = random.nextInt(10) + 1;

            Genotype<IntegerGene> genotype = Genotype.of(
                    IntegerChromosome.of(IntRange.of(1, 100), learningRate),
                    IntegerChromosome.of(IntRange.of(16, 128), batchSize),
                    IntegerChromosome.of(IntRange.of(1, 100), numberEpochs),
                    IntegerChromosome.of(IntRange.of(1, 20), earlyStoppingPatience),
                    IntegerChromosome.of(IntRange.of(1, 10), numberOfFineTuningLayers)
            );

            Phenotype<IntegerGene, Double> phenotype = Phenotype.of(genotype, 1);

            allIndividuals.add(new EvaluatedPhenotype(phenotype, null));
        }

        System.out.println("Population initialization complete. Population size: " + allIndividuals.size());
    }

    /**
     * Calculates the fitness for a given set of hyperparameters by sending them to the evaluation edge.
     *
     * @param learningRate             The learning rate.
     * @param batchSize                The batch size.
     * @param numberEpochs             The number of epochs.
     * @param earlyStoppingPatience    The early stopping patience.
     * @param numberOfFineTuningLayers The number of fine-tuning layers.
     * @return The fitness value representing the performance of the model.
     */
    private synchronized Double computeFitness(Double learningRate, Integer batchSize, Integer numberEpochs, Integer earlyStoppingPatience,
                                               Integer numberOfFineTuningLayers) {
        if (evaluationEdge == null) {
            throw new IllegalStateException("Evaluation edge is not set. Please set it before evaluating.");
        }

        System.out.println("Evaluating fitness for edge: " + evaluationEdge.getName());
        System.out.println("Fitness function with learning rate: " + learningRate + ", batch size: " + batchSize + ", " +
                "epochs: " + numberEpochs + ", patience: " + earlyStoppingPatience + ", fine tune layers: " +
                numberOfFineTuningLayers);

        try {
            // Send the model to the evaluation edge and get the performance
            Double fitnessValue =
                    modelFileHandler.sendFogModelToEdgesForGenetics(
                            evaluationEdge,
                            modelFileHandler.getFogModelPath(),
                            date,
                            learningRate,
                            batchSize,
                            numberEpochs,
                            earlyStoppingPatience,
                            numberOfFineTuningLayers
                    );

            if (fitnessValue == null) {
                System.out.println("Fitness evaluation returned null for parameters: learningRate=" + learningRate +
                        ", batchSize=" + batchSize + ", epochs=" + numberEpochs + ", patience=" + earlyStoppingPatience +
                        ", fineTuningLayers=" + numberOfFineTuningLayers);
                return Double.MAX_VALUE; // Default fitness value
            }

            System.out.println("Fitness value:" + fitnessValue);
            return fitnessValue;
        } catch (Exception e) {
            System.err.println("Exception occurred while sending fog model to evaluation edge. Error: " +
                    e.getMessage());
            return Double.MAX_VALUE; // Default performance value in case of error
        }
    }

    /**
     * Evaluates the fitness of a phenotype represented by the given parameter values.
     *
     * @param values The parameter values for evaluation: [learningRate, batchSize, epochs, patience, fineTuneLayers].
     * @return The calculated fitness value.
     */
    public Double evaluate(int[] values) {
        double learningRate = values[0] / 10000.0;
        int batchSize = values[1];
        int numberEpochs = values[2];
        int earlyStoppingPatience = values[3];
        int numberOfFineTuningLayers = values[4];

        Double fitness = computeFitness(learningRate, batchSize, numberEpochs, earlyStoppingPatience, numberOfFineTuningLayers);

        if (fitness == null) {
            System.err.println("Fitness evaluation returned null for values: " + Arrays.toString(values));
            return Double.MAX_VALUE;
        }

        // Update the corresponding EvaluatedPhenotype with the evaluated fitness
        Phenotype<IntegerGene, Double> phenotype = findPhenotypeInPopulation(values);
        if (phenotype != null) {
            // Find the corresponding EvaluatedPhenotype object
            for (EvaluatedPhenotype evaluatedPhenotype : allIndividuals) {
                if (evaluatedPhenotype.getPhenotype().equals(phenotype)) {
                    evaluatedPhenotype.setLastFitness(fitness);
                    break;
                }
            }
        }
        return fitness;
    }

    /**
     * Performs conditional evaluation of fitness by checking the cache or re-evaluating as needed.
     *
     * @param values The parameter values for evaluation.
     * @return The fitness value.
     */
    private Double conditionalEvaluate(int[] values) {
        System.out.println("Running conditional evaluation for values: " + Arrays.toString(values));

        Phenotype<IntegerGene, Double> phenotype = findPhenotypeInPopulation(values);
        if (phenotype == null) {
            System.out.println("Phenotype not found, creating new one.");
            phenotype = createPhenotype(values);
        } else {
            System.out.println("Phenotype found: " + phenotype);
        }

        GenotypeKey genotypeKey = new GenotypeKey(phenotype.genotype());
        Double cachedFitness = fitnessCache.get(genotypeKey);

        if (cachedFitness != null) {
            System.out.println("Cached fitness found: " + cachedFitness);
            if (shouldSkipCondition(cachedFitness)) {
                System.out.println("Skipping evaluation based on cached fitness: " + cachedFitness);
                return cachedFitness;
            }
        }

        Double newFitness = evaluate(values);
        fitnessCache.put(genotypeKey, newFitness);

        System.out.println("New fitness evaluated: " + newFitness + " for values: " + Arrays.toString(values));
        return newFitness;
    }

    /**
     * Finds a {@link Phenotype} in the current population that matches the given parameter values.
     *
     * @param values The parameter values to find in the population.
     * @return The matching phenotype, or {@code null} if not found.
     */
    private Phenotype<IntegerGene, Double> findPhenotypeInPopulation(int[] values) {
        for (EvaluatedPhenotype evalPhenotype : allIndividuals) {
            int[] existingValues = evalPhenotype.getPhenotype().genotype().stream()
                    .flatMap(BaseSeq::stream)
                    .mapToInt(IntegerGene::intValue)
                    .toArray();

            if (Arrays.equals(existingValues, values)) {
                return evalPhenotype.getPhenotype();
            }
        }
        return null;
    }

    /**
     * Determines whether a fitness evaluation should be skipped based on Boltzmann distribution.
     *
     * @param currentGenotypeFitness The current fitness value.
     * @return {@code true} if the evaluation should be skipped; otherwise {@code false}.
     */
    private Boolean shouldSkipCondition(Double currentGenotypeFitness) {
        System.out.println("Checking skip condition for fitness: " + currentGenotypeFitness);
        Double cloudTemperature = cloudCoolingSchedule.getCloudCoolingTemperature();
        System.out.println("cloud temp:" + cloudTemperature);
        // Boltzmann distribution vs random number
        return Math.random() > Math.exp((-currentGenotypeFitness) / (BOLTZMANN_CONSTANT * ADDITIONAL_FACTOR *
                cloudTemperature));
    }

    /**
     * Creates a new phenotype from the given parameter values.
     *
     * @param values The parameter values.
     * @return The created phenotype.
     */
    private Phenotype<IntegerGene, Double> createPhenotype(int[] values) {
        Genotype<IntegerGene> genotype = Genotype.of(
                IntegerChromosome.of(IntRange.of(1, 100), values[0]),
                IntegerChromosome.of(IntRange.of(16, 128), values[1]),
                IntegerChromosome.of(IntRange.of(1, 100), values[2]),
                IntegerChromosome.of(IntRange.of(1, 20), values[3]),
                IntegerChromosome.of(IntRange.of(1, 10), values[4])
        );
        return Phenotype.of(genotype, 1);
    }

    /**
     * Updates the population after each generation based on fitness values.
     *
     * @param result The evolution result containing the current population.
     */
    private void updatePopulation(EvolutionResult<IntegerGene, Double> result) {
        // Get the current population
        ISeq<Phenotype<IntegerGene, Double>> population = result.population();

        // Filter and sort individuals with non-null fitness
        List<EvaluatedPhenotype> bestIndividuals = allIndividuals.stream()
                .filter(individual -> individual.getLastFitness() != null)
                .sorted(Comparator.comparingDouble(EvaluatedPhenotype::getLastFitness))
                .limit(POPULATION_SIZE)
                .toList();

        // Update the allIndividuals list
        allIndividuals.clear();
        population.forEach(phenotype ->
                allIndividuals.add(new EvaluatedPhenotype(phenotype, phenotype.fitness())));

        // Merge with the best individuals
        allIndividuals.addAll(bestIndividuals);

        // Ensure the size remain fixed
        allIndividuals.sort(Comparator.comparingDouble(EvaluatedPhenotype::getLastFitness));
        if (allIndividuals.size() > POPULATION_SIZE) {
            allIndividuals.subList(POPULATION_SIZE, allIndividuals.size()).clear();
        }
    }

    /**
     * Executes the genetic algorithm to optimize hyperparameters.
     *
     * @param currentDate               The current date for the simulation.
     * @param geneticEvaluationStrategy The evaluation strategy for genetic evolution.
     */
    public void doGenetics(List<String> currentDate, GeneticEvaluationStrategy geneticEvaluationStrategy) {
        System.out.println("Starting genetic algorithm for date " + currentDate);
        this.date = currentDate;

        final InvertibleCodec<int[], IntegerGene> codec = Codecs.ofVector(
                IntRange.of(1, 100),        // learningRate range (scaled to [0.001, 0.1])
                IntRange.of(16, 128),       // batchSize range
                IntRange.of(1, 100),        // numberEpochs range
                IntRange.of(1, 20),         // earlyStoppingPatience range
                IntRange.of(1, 10)          // numberOfFineTuningLayers range
        );

        Engine.Builder<IntegerGene, Double> engineBuilder = Engine.builder(this::conditionalEvaluate, codec)
                .populationSize(POPULATION_SIZE)
                .offspringFraction(0.7)
                .survivorsSelector(new EliteSelector<>())
                .offspringSelector(new TournamentSelector<>(3))
                .alterers(new Mutator<>(0.1), new SinglePointCrossover<>(0.6))
                .minimizing();

        switch (geneticEvaluationStrategy) {
            case UPDATE_BOTTOM_INDIVIDUALS:
                System.out.println("Using top preserving interceptor...");
                engineBuilder.interceptor(new TopPreservingInterceptor<>(TOP_PRESERVE_COUNT, allIndividuals, this));
                break;
            case UPDATE_ALL_INDIVIDUALS:
                System.out.println("Using update all individuals interceptor...");
                InterceptorUtils<IntegerGene, Double> utilsForAllIndividuals = new InterceptorUtils<>(this);
                utilsForAllIndividuals.setPopulation(new ArrayList<>(allIndividuals.stream()
                        .map(EvaluatedPhenotype::getPhenotype)
                        .toList()));
                engineBuilder.interceptor(new UpdateAllIndividualsInterceptor(
                        allIndividuals, fitnessCache, this::evaluate, date.toString(), utilsForAllIndividuals
                ));
                break;
            default:
                Random random = new Random();
                if (random.nextBoolean()) {
                    System.out.println("Using top preserving interceptor from random...");
                    engineBuilder.interceptor(new TopPreservingInterceptor<>(TOP_PRESERVE_COUNT, allIndividuals, this));
                } else {
                    System.out.println("Using update all individuals interceptor from random...");
                    InterceptorUtils<IntegerGene, Double> utilsForRandom = new InterceptorUtils<>(this);
                    utilsForRandom.setPopulation(new ArrayList<>(allIndividuals.stream()
                            .map(EvaluatedPhenotype::getPhenotype)
                            .toList()));
                    engineBuilder.interceptor(new UpdateAllIndividualsInterceptor(
                            allIndividuals, fitnessCache, this::evaluate, date.toString(), utilsForRandom
                    ));
                }
                break;
        }

        Engine<IntegerGene, Double> engine = engineBuilder.build();

        EvolutionResult<IntegerGene, Double> result = engine.stream()
                .limit(NUMBER_GENERATIONS)
                .peek(this::logPopulation)
                .peek(this::updatePopulation)
                .takeWhile(r -> !isConverged(r))
                .collect(EvolutionResult.toBestEvolutionResult());

        System.out.println("Best solution for date " + currentDate + ": " + result.bestPhenotype());
    }

    /**
     * Checks whether the genetic algorithm has converged based on the stagnation in fitness values.
     *
     * @param result The evolution result.
     * @return {@code true} if the algorithm has converged; otherwise {@code false}.
     */
    private boolean isConverged(EvolutionResult<IntegerGene, Double> result) {
        double currentBestFitness = result.bestFitness();
        if (Math.abs(currentBestFitness - bestFitness) < 1e-6) {
            stagnationCounter++;
            if (stagnationCounter >= MAX_STAGNATION) {
                System.out.println("Convergence reached, stopping evolution.");
                return true;
            }
        } else {
            stagnationCounter = 0;
        }
        bestFitness = currentBestFitness;
        return false;
    }

    /**
     * Retrieves the top 3 individuals from the population based on their fitness values.
     *
     * @return A list of parameter arrays for the top 3 individuals.
     */
    public List<int[]> getTop3Individuals() {
        List<EvaluatedPhenotype> validIndividuals = allIndividuals.stream()
                .filter(individual -> individual.getLastFitness() != null && individual.getLastFitness() > 0)
                .sorted(Comparator.comparingDouble(EvaluatedPhenotype::getLastFitness))
                .toList();

        if (validIndividuals.isEmpty()) {
            System.err.println("ERROR: No valid individuals with fitness > 0 found.");
            return Collections.emptyList();
        }

        // Return top 3 individuals
        List<int[]> top3Params = new ArrayList<>();
        for (int i = 0; i < Math.min(3, validIndividuals.size()); i++) {
            Phenotype<IntegerGene, Double> phenotype = validIndividuals.get(i).getPhenotype();
            int[] params = phenotype.genotype().stream()
                    .flatMap(BaseSeq::stream)
                    .mapToInt(IntegerGene::intValue)
                    .toArray();
            top3Params.add(params);
        }

        return top3Params;
    }

    /**
     * Saves the state of the genetic algorithm, including the population and fitness cache.
     */
    public void saveState() {
        Gson gson = createGeneticGson();

        try (FileWriter individualWriter = new FileWriter(INDIVIDUALS_FILE);
             FileWriter fitnessCacheWriter = new FileWriter(FITNESS_CACHE_FILE)) {

            Type individualsListType = new TypeToken<List<EvaluatedPhenotype>>() {
            }.getType();
            gson.toJson(allIndividuals, individualsListType, individualWriter);

            // Serialize the list of EvaluatedPhenotype
            FitnessCacheManager.saveFitnessCache(allIndividuals, INDIVIDUALS_FILE);

            // Serialize the fitness cache using the DTO conversion
            List<EvaluatedGenotype> dtoList = new ArrayList<>();
            for (Map.Entry<GenotypeKey, Double> entry : fitnessCache.entrySet()) {
                UnevaluatedGenotype unevaluatedGenotype = GenotypeConverter.toDTO(entry.getKey().genotype());
                EvaluatedGenotype evaluatedGenotype = new EvaluatedGenotype(unevaluatedGenotype, entry.getValue());
                dtoList.add(evaluatedGenotype);
            }

            // Prepare the JSON structure
            JsonObject jsonObject = new JsonObject();
            jsonObject.add("genotypes", gson.toJsonTree(dtoList));

            // Write the JSON object to the file using the custom Gson instance
            gson.toJson(jsonObject, fitnessCacheWriter);

            System.out.println("Genetic state saved successfully!");

        } catch (IOException e) {
            System.err.println("Error saving genetic state: " + e.getMessage());
            throw new RuntimeException(e);
        }
    }

    /**
     * Loads the state of the genetic algorithm from saved files.
     */
    public void loadState() {
        Gson gson = createGeneticGson();

        try (FileReader individualReader = new FileReader(INDIVIDUALS_FILE);
             FileReader fitnessCacheReader = new FileReader(FITNESS_CACHE_FILE)) {

            char[] buffer = new char[4096];
            int len;
            StringBuilder jsonContent = new StringBuilder();
            while ((len = individualReader.read(buffer)) != -1) {
                jsonContent.append(buffer, 0, len);
            }

            Type individualsListType = new TypeToken<List<EvaluatedPhenotype>>() {
            }.getType();
            List<EvaluatedPhenotype> loadedIndividuals = gson.fromJson(jsonContent.toString(), individualsListType);

            StringBuilder fitnessJsonContent = new StringBuilder();
            while ((len = fitnessCacheReader.read(buffer)) != -1) {
                fitnessJsonContent.append(buffer, 0, len);
            }

            Type fitnessCacheType = new TypeToken<Map<GenotypeKey, Double>>() {
            }.getType();
            Map<GenotypeKey, Double> loadedCache = gson.fromJson(fitnessJsonContent.toString(), fitnessCacheType);

            allIndividuals.clear();
            fitnessCache.clear();

            allIndividuals.addAll(loadedIndividuals);
            fitnessCache.putAll(loadedCache);

            System.out.println("State loaded successfully!");

        } catch (IOException e) {
            System.out.println("Genetic state JSON file not found yet.");
        } catch (JsonSyntaxException e) {
            System.err.println("Failed to load genetic state: " + e.getMessage());
        }
    }

    /**
     * Creates a Gson instance with custom adapters for serializing/deserializing genetic types.
     *
     * @return A configured Gson instance.
     */
    public static Gson createGeneticGson() {
        return new GsonBuilder()
                .registerTypeAdapter(GenotypeKey.class, new GenotypeKeyAdapter())
                .registerTypeAdapter(new TypeToken<ISeq<Chromosome<IntegerGene>>>() {
                        }.getType(),
                        new ISeqChromosomeAdapter())
                .registerTypeAdapter(new TypeToken<ISeq<Phenotype<IntegerGene, Double>>>() {
                        }.getType(),
                        new ISeqTypeAdapter(null))
                .registerTypeAdapter(Chromosome.class, new ChromosomeAdapter())
                .registerTypeAdapter(Phenotype.class, new PhenotypeTypeAdapter())
                .registerTypeAdapter(new TypeToken<Map<GenotypeKey, Double>>() {
                        }.getType(),
                        new FitnessCacheDeserializer())
                .create();
    }

    /**
     * Logs the current population to the console for debugging purposes.
     *
     * @param result The evolution result containing the current population.
     */
    private void logPopulation(EvolutionResult<IntegerGene, Double> result) {
        ISeq<Phenotype<IntegerGene, Double>> population = result.population();
        System.out.println("Population size after evolution: " + (population != null ? population.size() : "null"));

        for (int i = 0; i < Objects.requireNonNull(population).size(); i++) {
            Phenotype<IntegerGene, Double> phenotype = population.get(i);
            Double fitness = phenotype.fitness();
            System.out.println("Phenotype " + i + ": " + phenotype + ", Fitness: " + (fitness != null ? fitness : "null"));
        }
    }
}
