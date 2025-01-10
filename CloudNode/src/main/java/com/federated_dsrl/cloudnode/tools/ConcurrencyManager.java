package com.federated_dsrl.cloudnode.tools;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import lombok.Data;
import org.springframework.stereotype.Component;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Manages concurrency mechanisms and state for the cloud node's operations.
 * <p>
 * This component handles thread pools, locks, and state persistence for aggregated dates
 * and elapsed times during iterations of federated learning processes.
 * </p>
 */
@Component
@Data
public class ConcurrencyManager {

    /**
     * Executor service for handling concurrent tasks with a fixed thread pool of size 3.
     */
    private final ExecutorService executorService = Executors.newFixedThreadPool(3);

    /**
     * Scheduler for periodic tasks with a single-threaded pool.
     */
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

    /**
     * Lock for ensuring exclusive access to shared resources during processing.
     */
    private final Lock receivingLock = new ReentrantLock();

    /**
     * Counter for tracking the number of print operations.
     */
    private final AtomicInteger printingCounter = new AtomicInteger();

    /**
     * Counter for tracking the number of times a resource has been found during processing.
     */
    private final AtomicInteger foundPrinterCounter = new AtomicInteger();

    /**
     * Maximum allowed number of print operations.
     */
    private final Integer allowedPrintingTimes = 5;

    /**
     * Counter for tracking the waiting operations during processing.
     */
    private final AtomicInteger waiterCounter = new AtomicInteger();

    /**
     * List of dates on which aggregation has been performed.
     */
    private List<String> aggregatedDates = new ArrayList<>();

    /**
     * Map for storing elapsed times for different iterations.
     */
    private Map<String, Double> elapsedTimeOverIterations = new HashMap<>();

    /**
     * File path for saving aggregated dates to a JSON file.
     */
    private final String AGGREGATED_DATES_FILE = "cache_json/aggregated_dates.json";

    /**
     * File path for saving elapsed times over iterations to a JSON file.
     */
    private final String ELAPSED_TIME_OVER_ITERATIONS_FILE = "cache_json/elapsed_time_over_iterations.json";

    /**
     * Flag to indicate if aggregation has already been performed.
     */
    private Boolean alreadyAggregated = Boolean.FALSE;

    /**
     * Clears the list of aggregated dates.
     */
    public void clearAggregatedDate() {
        aggregatedDates.clear();
    }

    /**
     * Clears the map of elapsed times over iterations.
     */
    public void clearElapsedTimeOverIterations() {
        elapsedTimeOverIterations.clear();
    }

    /**
     * Retrieves the most recent aggregated date.
     *
     * @return the last aggregated date, or {@code null} if the list is empty.
     */
    public String getLastAggregatedDate() {
        return aggregatedDates.isEmpty() ? null : aggregatedDates.get(aggregatedDates.size() - 1);
    }

    /**
     * Adds a new aggregated date to the list.
     *
     * @param aggregatedDate the date to add.
     */
    public void addNewAggregatedDate(String aggregatedDate) {
        this.aggregatedDates.add(aggregatedDate);
    }

    /**
     * Saves the current state of aggregated dates and elapsed times to JSON files.
     */
    public void saveCacheToJsonFile() {
        Gson gson = new Gson();
        try (FileWriter aggregatedWriter = new FileWriter(AGGREGATED_DATES_FILE);
             FileWriter elapsedWriter = new FileWriter(ELAPSED_TIME_OVER_ITERATIONS_FILE)) {
            gson.toJson(aggregatedDates, aggregatedWriter);
            gson.toJson(elapsedTimeOverIterations, elapsedWriter);
            System.out.println("Successfully written to JSON: aggregated dates and elapsed time over iterations.");
        } catch (IOException e) {
            System.out.println("Error writing to JSON: aggregated dates and elapsed time over iterations.");
            throw new RuntimeException(e);
        }
    }

    /**
     * Loads the state of aggregated dates and elapsed times from JSON files.
     */
    public void loadCacheFromJsonFile() {
        Gson gson = new Gson();
        try (FileReader aggregatedReader = new FileReader(AGGREGATED_DATES_FILE);
             FileReader elapsedReader = new FileReader(ELAPSED_TIME_OVER_ITERATIONS_FILE)) {
            Type aggregatedType = new TypeToken<List<String>>() {}.getType();
            Type elapsedType = new TypeToken<Map<String, Double>>() {}.getType();

            List<String> aggregatedDatesList = gson.fromJson(aggregatedReader, aggregatedType);
            Map<String, Double> elapsedTimeOverIterationsMap = gson.fromJson(elapsedReader, elapsedType);

            aggregatedDates.clear();
            elapsedTimeOverIterations.clear();
            aggregatedDates.addAll(aggregatedDatesList);
            elapsedTimeOverIterations.putAll(elapsedTimeOverIterationsMap);
            System.out.println("Successfully loaded from JSON: aggregated dates and elapsed time over iterations.");
        } catch (IOException e) {
            System.out.println("Could not load from JSON: aggregated dates and elapsed time over iterations.");
        }
    }
}
