package com.federated_dsrl.fognode.tools;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.federated_dsrl.fognode.entity.ElapsedTimeOneFogIteration;
import lombok.Data;
import org.springframework.stereotype.Component;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Manages concurrency and scheduling for the fog node system.
 * <p>
 * This class provides a range of features, including:
 * <ul>
 *     <li>Task scheduling using {@link ScheduledExecutorService} and {@link ExecutorService}</li>
 *     <li>Locking mechanisms for synchronized operations</li>
 *     <li>Management of elapsed time data for fog iterations</li>
 *     <li>Retry mechanisms for tasks</li>
 * </ul>
 * </p>
 */
@Component
@Data
public class ConcurrencyManager {

    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(5);
    private final ExecutorService executorService = new ThreadPoolExecutor(
            30, // Core pool size
            70, // Maximum pool size
            60L, TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(150), // Bounded queue with capacity 150
            new ThreadPoolExecutor.CallerRunsPolicy() // Rejection policy
    );
    private final Lock receivingLock = new ReentrantLock();
    private Boolean atLeastOneAggregation = Boolean.FALSE;
    private final AtomicInteger printerCounter = new AtomicInteger();
    private final Integer allowedPrintingTimes = 5;
    private String currentDate;
    private final int MAX_RETRY_ATTEMPTS = 5;
    private final int RETRY_DELAY_MS = 2000;
    private final List<ElapsedTimeOneFogIteration> elapsedTimeOneFogIterationList = new ArrayList<>();
    private ElapsedTimeOneFogIteration elapsedTimeOneFogIteration;
    private final String ELAPSED_TIME_FILE = "cache_json/elapsed_time.json";

    // Prevent repeated sharing of fog models
    private Boolean alreadySentFogModel = Boolean.FALSE;

    /**
     * Acquires the lock to synchronize operations.
     */
    public void lock() {
        receivingLock.lock();
    }

    /**
     * Releases the lock after synchronized operations are complete.
     */
    public void unlock() {
        receivingLock.unlock();
    }

    /**
     * Initializes the elapsed time manager for the current fog iteration.
     * <p>
     * This method sets the initial elapsed time for genetic evaluation to 0.0.
     * </p>
     */
    public void initCurrentElapsedTimeManager() {
        this.elapsedTimeOneFogIteration = new ElapsedTimeOneFogIteration();
        this.elapsedTimeOneFogIteration.setTimeGeneticEvaluation(0.0);
    }

    /**
     * Adds the current elapsed time manager to the list of elapsed times for fog iterations.
     */
    public void addElapsedTimeManagerToList() {
        this.elapsedTimeOneFogIterationList.add(elapsedTimeOneFogIteration);
    }

    /**
     * Saves the list of elapsed times for fog iterations to a JSON file.
     * <p>
     * The file is stored at the location defined by {@code ELAPSED_TIME_FILE}.
     * </p>
     */
    public void saveElapsedTimeToJsonFile() {
        Gson gson = new Gson();

        try (FileWriter elapsedTimeWriter = new FileWriter(ELAPSED_TIME_FILE)) {
            gson.toJson(elapsedTimeOneFogIterationList, elapsedTimeWriter);
            System.out.println("Elapsed time one fog iteration saved as JSON successfully.");
        } catch (IOException e) {
            System.out.println("Error saving elapsed time one fog iteration.");
            throw new RuntimeException(e);
        }
    }

    /**
     * Loads the list of elapsed times for fog iterations from a JSON file.
     * <p>
     * Any duplicate elapsed times are ignored during the loading process.
     * </p>
     */
    public void loadElapsedTimeToJsonFile() {
        Gson gson = new Gson();

        try (FileReader elapsedTimeReader = new FileReader(ELAPSED_TIME_FILE)) {
            Type elapsedTimeType = new TypeToken<List<ElapsedTimeOneFogIteration>>() {
            }.getType();
            List<ElapsedTimeOneFogIteration> elapsedTime = gson.fromJson(elapsedTimeReader, elapsedTimeType);
            elapsedTime.forEach(time -> {
                if (!containsTime(time.getTimeGeneticEvaluation())) {
                    ElapsedTimeOneFogIteration elapsedTimeOneFogIteration1 = new ElapsedTimeOneFogIteration();
                    elapsedTimeOneFogIteration1.setTimeGeneticEvaluation(time.getTimeGeneticEvaluation());
                    elapsedTimeOneFogIterationList.add(elapsedTimeOneFogIteration1);
                }
            });
            System.out.println("Elapsed time over fog iterations loaded successfully.");
        } catch (IOException e) {
            System.out.println("Elapsed time over fog iteration JSON file was not found yet.");
        }
    }

    /**
     * Checks if a specific elapsed time already exists in the list of fog iteration times.
     *
     * @param time The elapsed time to check.
     * @return {@code true} if the elapsed time is present; otherwise {@code false}.
     */
    private Boolean containsTime(Double time) {
        return elapsedTimeOneFogIterationList.stream()
                .map(ElapsedTimeOneFogIteration::getTimeGeneticEvaluation)
                .toList()
                .contains(time);
    }
}
