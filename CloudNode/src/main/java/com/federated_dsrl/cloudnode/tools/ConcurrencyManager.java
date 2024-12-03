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

@Component
@Data
public class ConcurrencyManager {
    private final ExecutorService executorService = Executors.newFixedThreadPool(3);
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
    private final Lock receivingLock = new ReentrantLock();
    private final AtomicInteger printingCounter = new AtomicInteger();
    private final AtomicInteger foundPrinterCounter = new AtomicInteger();
    private final Integer allowedPrintingTimes = 5;
    private final AtomicInteger waiterCounter = new AtomicInteger();
    private List<String> aggregatedDates = new ArrayList<>();
    private Map<String, Double> elapsedTimeOverIterations = new HashMap<>();
    private final String AGGREGATED_DATES_FILE = "cache_json/aggregated_dates.json";
    private final String ELAPSED_TIME_OVER_ITERATIONS_FILE = "cache_json/elapsed_time_over_iterations.json";
    private Boolean alreadyAggregated = Boolean.FALSE;

    public void clearAggregatedDate() {
        aggregatedDates.clear();
    }

    public void clearElapsedTimeOverIterations() {
        elapsedTimeOverIterations.clear();
    }

    public String getLastAggregatedDate() {
        return aggregatedDates.isEmpty() ? null : aggregatedDates.get(aggregatedDates.size() - 1);
    }

    public void addNewAggregatedDate(String aggregatedDate) {
        this.aggregatedDates.add(aggregatedDate);
    }

    public void saveCacheToJsonFile() {
        Gson gson = new Gson();

        try (FileWriter aggregatedWriter = new FileWriter(AGGREGATED_DATES_FILE);
             FileWriter elapsedWriter = new FileWriter(ELAPSED_TIME_OVER_ITERATIONS_FILE)) {
            gson.toJson(aggregatedDates, aggregatedWriter);
            gson.toJson(elapsedTimeOverIterations, elapsedWriter);
            System.out.println("Successfully written to json aggregated dates and elapsed time over iterations.");
        } catch (IOException e) {
            System.out.println("Error writing to json aggregated dates and elapsed time over iterations.");
            throw new RuntimeException(e);
        }
    }

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
            System.out.println("Successfully loaded from json aggregated dates and elapsed time over iterations.");
        } catch (IOException e) {
            System.out.println("Not yet loading from json aggregated dates and elapsed time over iterations.");
        }
    }
}
