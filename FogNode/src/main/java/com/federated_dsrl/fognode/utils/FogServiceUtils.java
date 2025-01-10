package com.federated_dsrl.fognode.utils;

import com.federated_dsrl.fognode.config.DeviceManager;
import com.federated_dsrl.fognode.config.PathManager;
import com.federated_dsrl.fognode.entity.FogTraffic;
import com.federated_dsrl.fognode.tools.ConcurrencyManager;
import com.federated_dsrl.fognode.config.AggregationType;
import com.federated_dsrl.fognode.config.EdgeReadinessManager;
import com.federated_dsrl.fognode.entity.EdgeEntity;
import com.federated_dsrl.fognode.tools.CountDownLatchManager;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Utility class for managing fog node services and interactions.
 * <p>
 * This class provides various methods for:
 * <ul>
 *     <li>Sending models to edges asynchronously</li>
 *     <li>Notifying edges about fog nodes</li>
 *     <li>Handling fog model transmission to the cloud</li>
 *     <li>Deleting temporary files and managing statistics</li>
 *     <li>Handling edge readiness and synchronization</li>
 * </ul>
 * </p>
 */
@Component
@RequiredArgsConstructor
public class FogServiceUtils {
    private static final int POLLING_INTERVAL_MS = 1000;
    private final ConcurrencyManager concurrencyManager;
    private final PathManager pathManager;
    private final DeviceManager deviceManager;
    private final RestTemplate restTemplate = new RestTemplate();
    private final ModelFileHandler modelFileHandler;
    private final EdgeReadinessManager edgeReadinessManager;
    private final FogTraffic fogTraffic;
    private final CountDownLatchManager latchManager;
    @Getter
    private final HttpHeaders jsonHeaders = createJsonHeaders();

    /**
     * Creates default JSON headers for HTTP requests.
     *
     * @return {@link HttpHeaders} configured for JSON content.
     */
    private HttpHeaders createJsonHeaders() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);
        return headers;
    }

    /**
     * Logs an informational message to the console in green.
     *
     * @param message The message to log.
     */
    public void logInfo(String message) {
        // ANSI escape code for green text
        String green = "\u001B[32m";
        // ANSI escape code for resetting to default color
        String reset = "\u001B[0m";
        System.out.println(green + "INFO: " + reset + message);
    }

    /**
     * Logs an error message to the console in red.
     *
     * @param message The message to log.
     */
    public void logError(String message) {
        // ANSI escape code for red text
        String red = "\u001B[31m";
        // ANSI escape code for resetting to default color
        String reset = "\u001B[0m";
        System.err.println(red + "ERROR: " + reset + message);
    }


    /**
     * Sets the training date for the fog node system and sends network configuration if initial training.
     *
     * @param date             The list of dates.
     * @param isInitialTraining {@code true} if this is the initial training.
     */
    public void setTrainingDate(List<String> date, Boolean isInitialTraining) {
        if (isInitialTraining) {
            // if initial, send network configuration info
            notifyEdgeAboutFog();
            concurrencyManager.setCurrentDate(date.get(1));
        } else {
            concurrencyManager.setCurrentDate(date.get(0));
        }
    }

    /**
     * Notifies edges about the fog node's host configuration.
     */
    public void notifyEdgeAboutFog() {
        String hostAddress = System.getenv("HOST_IP");
        logInfo("Host address from environment variable: " + hostAddress);
        String lastByte = hostAddress.split("\\.")[3];

        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        body.add("host", lastByte);

        HttpEntity<MultiValueMap<String, String>> requestEntity = new HttpEntity<>(body, jsonHeaders);

        deviceManager.getEdges().forEach(edge -> {
            String endpoint = edge.getEndpoints().get("host");
            logInfo("Sending request to endpoint: " + endpoint + " with body: " + body);
            sendPostRequest(endpoint, requestEntity);
        });
    }

    /**
     * Sends a POST request to a specified URL with the given request entity.
     *
     * @param url           The URL to send the request to.
     * @param requestEntity The request body and headers.
     */
    public void sendPostRequest(String url, HttpEntity<?> requestEntity) {
        try {
            ResponseEntity<String> response = restTemplate.postForEntity(url, requestEntity, String.class);
            if (response.getStatusCode().is2xxSuccessful()) {
                logInfo("Successfully notified endpoint: " + url);
            } else {
                logError("Failed to notify endpoint: " + url + ". Status code: " + response.getStatusCode());
            }
        } catch (Exception e) {
            logError("Exception while notifying endpoint: " + url + ". Error: " + e.getMessage());
        }
    }

    /**
     * Sends a model asynchronously to an edge.
     *
     * @param edge             The edge entity to send the model to.
     * @param fogModelPath     The path to the fog model file.
     * @param date             The list of dates for training.
     * @param isInitialTraining {@code true} if this is the initial training.
     * @param aggregationType  The type of aggregation being used.
     */
    public void sendModelToEdgeAsync(EdgeEntity edge, Path fogModelPath, List<String> date, Boolean isInitialTraining,
                                     AggregationType aggregationType) {
        CompletableFuture.runAsync(() -> {
            logInfo("Submitting task to send model to edge: " + edge.getName());
            try {
                modelFileHandler.sendCloudModelToEdgesWithRetry(edge, fogModelPath, date, isInitialTraining,
                        aggregationType);
            } catch (IOException e) {
                logError("Exception occurred while sending cloud model to: " + edge.getName() + ": " + e.getMessage());
            }
        }, concurrencyManager.getExecutorService());
    }

    /**
     * Asynchronously waits for an edge to become ready.
     *
     * @param edge The edge entity to wait for.
     */
    public void waitForEdgeReadinessAsync(EdgeEntity edge) {
        CompletableFuture.runAsync(() -> {
            logInfo("Submitting task to wait for readiness of edge: " + edge.getName());
            try {
                waitForEdgeReadiness(edge);
            } catch (InterruptedException e) {
                logError("InterruptedException occurred while waiting for edge readiness: " + edge.getName() + ": " + e.getMessage());
                throw new RuntimeException(e);
            }
            latchManager.countDown(edge.getLclid()); // Count down after waiting for readiness
            logInfo("Latch count after edge " + edge.getName() + " is ready: " + latchManager.getLatchCount());
        }, concurrencyManager.getExecutorService());
    }

    /**
     * Waits for an edge to become ready by polling its readiness state.
     *
     * @param edge The edge entity to wait for.
     * @throws InterruptedException If the thread is interrupted while waiting.
     */
    private void waitForEdgeReadiness(EdgeEntity edge) throws InterruptedException {
        logInfo("Start waiting for edge " + edge.getName());
        while (!edgeReadinessManager.isEdgeReady(edge.getLclid())) {
            logInfo("Waiting for edge " + edge.getName() + " to be ready...");
            Thread.sleep(POLLING_INTERVAL_MS); // Polling interval
        }
        logInfo("Edge " + edge.getName() + " is now ready.");
    }

    /**
     * Receives a readiness signal from an edge.
     *
     * @param lclid The LCID (Logical Cluster Identifier) of the edge.
     */
    public void receiveEdgeReadinessSignal(String lclid) {
        logInfo("Received readiness signal from edge with LCID: " + lclid);

        // check if the edge is already marked as ready, only count down if it's not.
        if (!latchManager.getCountedEdges().contains(lclid)) {
            edgeReadinessManager.markEdgeAsReady(lclid);
            latchManager.countDown(lclid);  // this will only count down once per edge
        } else {
            logInfo("Edge " + lclid + " has already been marked as ready, skipping count down.");
        }
    }

    /**
     * Determines whether an edge should proceed with an operation.
     *
     * @param lclid The LCID of the edge.
     * @return A {@link ResponseEntity} containing the result as a boolean.
     */
    public ResponseEntity<?> shouldEdgeProceed(String lclid) {
        boolean proceed = edgeReadinessManager.shouldProceed(lclid);
        return ResponseEntity.ok(proceed);
    }

    /**
     * Sends a fog model to the cloud, including additional metadata if required.
     *
     * @param aggregationType The type of aggregation used.
     */
    public void sendFogModelToCloud(AggregationType aggregationType) {
        try {
            concurrencyManager.addElapsedTimeManagerToList();
            System.out.printf("Sending fog aggregated model to cloud after "  + aggregationType + " strategy.");
            Path fogModelPath = Paths.get(pathManager.getModelsDirectory(), pathManager.getFogModelPath());
            File fogModelFile = fogModelPath.toFile();
            String lambdaPrev;

            if (aggregationType.equals(AggregationType.GENETIC)) {
                Path lambdaPrevPath = Paths.get(pathManager.getModelsDirectory(), "lambda_prev.json");
                try (BufferedReader reader = new BufferedReader(new FileReader(lambdaPrevPath.toFile()))) {
                    lambdaPrev = reader.readLine();
                }
            } else {
                lambdaPrev = null;
            }

            RestTemplate restTemplate = new RestTemplate();
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("result", lambdaPrev);
            body.add("current_date", concurrencyManager.getCurrentDate());
            body.add("fog_name", deviceManager.getFogName());
            body.add("file", new FileSystemResource(fogModelFile));

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
            try {
                ResponseEntity<String> response = restTemplate.postForEntity("http://192.168.2." +
                        deviceManager.getCloudHost() + ":8080/cloud/receive-fog-model", requestEntity, String.class);

                if (response.getStatusCode().is2xxSuccessful()) {
                    System.out.println("Successfully sent fog model to the cloud.");
                } else {
                    System.out.println("Failed to send fog model to the cloud. Status code: " +
                            response.getStatusCode());
                }
            } catch (Exception e) {
                System.out.println("Exception while sending fog model to the cloud: " + e.getMessage());
            }
        } catch (IOException e) {
            System.out.println("Exception while sending fog model to the cloud: " + e.getMessage());
        }
    }

    /**
     * Deletes temporary model files from the local directory.
     */
    public void deleteEdgeModelFiles() {
        Path directoryPath = Paths.get(pathManager.getModelsDirectory()).toAbsolutePath();
        File directory = new File(String.valueOf(directoryPath));
        if (!directory.exists() || !directory.isDirectory()) {
            System.err.println("Invalid directory, unable to delete fog model files.");
            return;
        }

        File[] fogModelFiles = directory.listFiles((dir, name) -> name.startsWith("edge_best_model") &&
                name.endsWith(".keras"));
        if (fogModelFiles == null || fogModelFiles.length == 0) {
            System.out.println("No fog model files found to delete.");
            return;
        }

        for (File fogModelFile : fogModelFiles) {
            if (fogModelFile.delete()) {
                System.out.println("Deleted fog model file: " + fogModelFile.getName());
            } else {
                System.err.println("Failed to delete fog model file: " + fogModelFile.getName());
            }
        }
    }

    /**
     * Deletes all JSON files in the cache directory.
     */
    public void deleteCacheJsonFolderContent() {
        File directory = new File("/app/cache_json");
        if (!directory.exists() || !directory.isDirectory()) {
            System.err.println("Invalid directory, unable to delete fog model files.");
            return;
        }

        File[] cacheJsonFiles = directory.listFiles((dir, name) -> name.endsWith(".json"));
        if (cacheJsonFiles == null || cacheJsonFiles.length == 0) {
            System.err.println("No json files found to delete.");
            return;
        }

        for (File jsonFile : cacheJsonFiles) {
            if (jsonFile.delete()) {
                System.out.printf("Deleted json file: " + jsonFile.getName());
            } else {
                System.err.println("Failed to delete json file:" + jsonFile.getName());
            }
        }
    }

    /**
     * Stores the traffic in local cache that the fog has registered this round to be able to analyze it later
     */
    public void statisticsHelper() {
        fogTraffic.storeCurrentIterationIncomingTraffic();
        fogTraffic.storeCurrentIterationOutgoingTraffic();
        fogTraffic.resetIncomingTraffic();
        fogTraffic.resetOutgoingTraffic();

        fogTraffic.saveTrafficToJsonFile();

        System.out.println("Fog incoming traffic: " + fogTraffic.getIncomingTrafficOverIterations());
        System.out.println("Fog outgoing traffic: " + fogTraffic.getOutgoingTrafficOverIterations());
    }

    /**
     * Caches the training date received from the cloud for processing and informs then the edges about it
     * @param dateList The list of date strings received from cloud
     */
    public void setEdgeWorkingDate(List<String> dateList) {
        String date = extractDate(dateList);
        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        body.add("date", date);
        HttpEntity<MultiValueMap<String, String>> requestEntity = new HttpEntity<>(body,
                getJsonHeaders());

        deviceManager.getEdges().forEach(edge -> {
            String endpoint = edge.getEndpoints().get("date");
            System.out.println(endpoint);
            sendPostRequest(endpoint, requestEntity);
        });
    }

    /**
     * Extracts a valid date from a list of strings.
     *
     * @param dateList The list of date strings.
     * @return The extracted date in "yyyy-MM-dd" format.
     * @throws IllegalArgumentException If no valid date is found in the list.
     */
    private String extractDate(List<String> dateList) {
        String dateStr = dateList.size() == 2 ? dateList.get(1) : dateList.get(0);
        Pattern pattern = Pattern.compile("\\d{4}-\\d{2}-\\d{2}");
        Matcher matcher = pattern.matcher(dateStr);
        if (matcher.find()) {
            return matcher.group(0);
        }
        throw new IllegalArgumentException("No valid date found in string: " + dateStr);
    }
}
