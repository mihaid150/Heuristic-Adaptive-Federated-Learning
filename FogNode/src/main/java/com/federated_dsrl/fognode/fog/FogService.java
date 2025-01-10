package com.federated_dsrl.fognode.fog;

import com.federated_dsrl.fognode.entity.FogTraffic;
import com.federated_dsrl.fognode.tools.ConcurrencyManager;
import com.federated_dsrl.fognode.tools.CountDownLatchManager;
import com.federated_dsrl.fognode.tools.simulated_annealing.FogCoolingSchedule;
import com.federated_dsrl.fognode.tools.simulated_annealing.MonitorCoolingSchedule;
import com.federated_dsrl.fognode.utils.ModelFileHandler;
import com.federated_dsrl.fognode.config.DeviceManager;
import com.federated_dsrl.fognode.config.EdgeReadinessManager;
import com.federated_dsrl.fognode.config.PathManager;
import com.federated_dsrl.fognode.entity.EdgeEntity;
import com.federated_dsrl.fognode.tools.genetic.engine.EdgeSelector;
import com.federated_dsrl.fognode.tools.genetic.engine.GeneticEngine;
import com.federated_dsrl.fognode.tools.genetic.interceptor.GeneticEvaluationStrategy;
import com.federated_dsrl.fognode.tools.simulated_annealing.CoolingScheduleState;
import com.federated_dsrl.fognode.config.AggregationType;
import com.federated_dsrl.fognode.utils.FogServiceUtils;
import lombok.RequiredArgsConstructor;
import org.apache.commons.lang3.time.StopWatch;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;

/**
 * Service class for managing fog node operations in the federated learning framework.
 * <p>
 * This class provides functionality for handling edge and cloud communication, training model
 * processing, genetic algorithm-based optimization, traffic monitoring, and system state management.
 * </p>
 */
@Service
@RequiredArgsConstructor
public class FogService {
    // Dependencies and utilities injected via constructor
    private final PathManager pathManager;
    private final DeviceManager deviceManager;
    private final ConcurrencyManager concurrencyManager;
    private final FogCoolingSchedule fogCoolingSchedule;
    private final EdgeReadinessManager edgeReadinessManager;
    private final MonitorCoolingSchedule monitorCoolingSchedule;
    private final GeneticEngine geneticEngine;
    private final ModelFileHandler modelFileHandler;
    private final EdgeSelector edgeSelector;
    private final StopWatch stopWatch;
    private final FogTraffic fogTraffic;
    private final FogServiceUtils fogServiceUtils;
    private final CountDownLatchManager latchManager;

    /**
     * Acknowledges the cloud node by setting its host address and name in the system.
     *
     * @param host the cloud's host address
     * @param name the cloud's name
     */
    public void ackCloud(String host, String name) {
        deviceManager.setCloudHost(host);
        deviceManager.setFogName(name);
    }

    /**
     * Adds a new edge to the fog node's device manager and configures its APIs.
     *
     * @param name  the edge's name
     * @param host  the edge's host address
     * @param lclid the local client ID (lclid) of the edge
     */
    public void addEdge(String name, String host, String lclid) {
        deviceManager.addEdge(name, host, lclid, fogServiceUtils);
    }

    /**
     * Notifies all associated edges about the current training model type.
     *
     * @param modelType the type of model (e.g., LSTM)
     */
    public void notifyEdgeAboutModelType(String modelType) {
        HttpHeaders headers = new org.springframework.http.HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);

        for(EdgeEntity edge: deviceManager.getEdges()) {
            RestTemplate restTemplate = new RestTemplate();
            MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
            body.add("model_type", modelType);
            HttpEntity<MultiValueMap<String, String>> requestEntity = new HttpEntity<>(body, headers);
            String url = edge.getEndpoints().get("model");
            ResponseEntity<String> response = restTemplate.postForEntity(url, requestEntity, String.class);
            if (response.getStatusCode().is2xxSuccessful()) {
                System.out.println("Successfully notified fog " + edge.getName() + " about edge model type.");
            } else {
                System.err.println("Unsuccessfully notified fog " + edge.getName() + " about edge model type."
                        + ". " + "Status code: " + response.getStatusCode());
            }
        }
    }

    /**
     * Receives the global model from the cloud and initiates edge training processes.
     *
     * @param modelFile                 the global model file
     * @param date                      the training iteration date
     * @param isInitialTraining         whether this is the first training iteration
     * @param isCacheActive             whether caching is enabled
     * @param geneticEvaluationStrategyArgument the genetic evaluation strategy to use
     * @throws IOException          if an error occurs during file operations
     * @throws InterruptedException if the latch wait is interrupted
     */
    public void receiveCloudModel(MultipartFile modelFile, List<String> date, Boolean isInitialTraining,
                                  Boolean isCacheActive, String geneticEvaluationStrategyArgument)
            throws IOException, InterruptedException {
        // Set edge working date and initialize genetic evaluation strategy
        fogServiceUtils.setEdgeWorkingDate(date);
        GeneticEvaluationStrategy geneticEvaluationStrategy = parseGeneticEvaluationStrategy(geneticEvaluationStrategyArgument);

        // reset and start the watch for elapsed time monitoring
        stopWatch.reset();
        stopWatch.start();
        if (isInitialTraining) {
            // clean the local storage from previous simulation cache
            fogServiceUtils.deleteCacheJsonFolderContent();
            fogServiceUtils.deleteEdgeModelFiles();
            fogTraffic.clearTraffic();

            // flag for initial aggregation
            concurrencyManager.setAtLeastOneAggregation(Boolean.FALSE);
        }

        // load the cache from previous iteration if not initial
        concurrencyManager.initCurrentElapsedTimeManager();
        fogTraffic.loadTrafficFromJsonFile();
        concurrencyManager.loadElapsedTimeToJsonFile();

        fogServiceUtils.logInfo("Receiving cloud model. Date: " + date + ", isInitialTraining: " + isInitialTraining);

        // validate the cloud model and save it locally
        modelFileHandler.validateModelFile(modelFile);
        modelFileHandler.saveFogModel(modelFile);

        // current fog training date set
        fogServiceUtils.setTrainingDate(date, isInitialTraining);

        edgeReadinessManager.resetReadiness(deviceManager.getEdges());
        edgeReadinessManager.initializeReadiness(deviceManager.getEdges());
        Path fogModelPath = modelFileHandler.getFogModelPath();

        edgeSelector.selectRandomEdgeAndGetRemaining();
        EdgeEntity evaluateEdge = edgeSelector.getEvaluateEdge();
        List<EdgeEntity> trainingEdges = edgeSelector.getTrainingEdges();
        System.out.println("training edges size: " + trainingEdges.size());

        geneticEngine.setEvaluationEdge(evaluateEdge);

        if (isInitialTraining) {
            geneticEngine.initializePopulation();
        }
        double startGeneticTime = stopWatch.getTime() / 1000.0;
        geneticEngine.doGenetics(date, geneticEvaluationStrategy);
        concurrencyManager.getElapsedTimeOneFogIteration()
                .setTimeGeneticEvaluation((stopWatch.getTime() / 1000.0) - startGeneticTime);
        concurrencyManager.saveElapsedTimeToJsonFile();
        concurrencyManager.loadElapsedTimeToJsonFile();
        notifyEdgesAboutTrainingParameters(trainingEdges);

        if (isCacheActive) {
            geneticEngine.saveState();
        }

        latchManager.resetLatch(trainingEdges.size());
        fogServiceUtils.logInfo("Initial latch count: " + latchManager.getLatchCount());

        for (EdgeEntity edge : trainingEdges) {
            fogServiceUtils.sendModelToEdgeAsync(edge, fogModelPath, date, isInitialTraining, AggregationType.GENETIC);
            fogServiceUtils.waitForEdgeReadinessAsync(edge);
        }

        fogServiceUtils.logInfo("Waiting for all edges to send model and become ready...");
        latchManager.await();

        if (stopWatch.isStarted()) {
            stopWatch.stop();
        }
        stopWatch.reset();
        fogServiceUtils.logInfo("All edges are ready. Starting cooling process.");

        stopWatch.start();
        fogCoolingSchedule.startFogCoolingScheduleThread();
        edgeReadinessManager.signalEdgesToProceed(trainingEdges);
    }

    /**
     * Parses the genetic evaluation strategy from a string.
     *
     * @param strategyArg the strategy argument
     * @return the parsed {@link GeneticEvaluationStrategy}
     */
    private GeneticEvaluationStrategy parseGeneticEvaluationStrategy(String strategyArg) {
        try {
            return GeneticEvaluationStrategy.valueOf(strategyArg.toUpperCase());
        } catch (IllegalArgumentException e) {
            fogServiceUtils.logError("Invalid genetic evaluation strategy: " + strategyArg);
            return GeneticEvaluationStrategy.UPDATE_BOTTOM_INDIVIDUALS;
        }
    }

    /**
     * Receives a readiness signal from an edge device.
     *
     * @param lclid the local client ID (lclid) of the edge
     */
    public void receiveEdgeReadinessSignal(String lclid) {
        fogServiceUtils.receiveEdgeReadinessSignal(lclid);
    }

    /**
     * Determines whether an edge device should proceed with its operation.
     *
     * @param lclid the local client ID (lclid) of the edge
     * @return a response indicating whether the edge should proceed
     */
    public ResponseEntity<?> shouldEdgeProceed(String lclid) {
        return fogServiceUtils.shouldEdgeProceed(lclid);
    }

    /**
     * Aggregates edge models using the genetic aggregation strategy.
     *
     * @param newEdgePerformance the performance of the new edge model
     * @param oldFogPerformance  the performance of the old fog model
     * @param lclid              the local client ID (lclid) of the edge
     * @param edgeModel          the edge model file
     * @throws IOException if an error occurs during file operations
     */
    public void receiveEdgeModel(Double newEdgePerformance, Double oldFogPerformance, String lclid,
                                 MultipartFile edgeModel) throws IOException {
        concurrencyManager.lock();
        try {
            fogServiceUtils.logInfo("Receiving edge model. New Edge Performance: " + newEdgePerformance +
                    ", Old Fog Performance: " + oldFogPerformance + ", LCID: " + lclid);
            modelFileHandler.validateModelFile(edgeModel);

            if (fogCoolingSchedule.getState().equals(CoolingScheduleState.OPERATIONAL)) {
                Path edgeModelPath = modelFileHandler.saveEdgeModel(edgeModel, lclid, fogServiceUtils);
                fogServiceUtils.logInfo("New edge model was saved to: " + edgeModelPath);

                if (fogCoolingSchedule.getState().equals(CoolingScheduleState.OPERATIONAL)) {
                    aggregateModels(newEdgePerformance, oldFogPerformance, edgeModelPath);
                    fogCoolingSchedule.incrementCounter();
                }
            }
        } finally {
            concurrencyManager.unlock();
        }
    }

    /**
     * Aggregates models using the genetic algorithm.
     *
     * @param newEdgePerformance the new edge model's performance
     * @param oldFogPerformance  the old fog model's performance
     * @param edgeModelPath      the path to the edge model file
     */
    private void aggregateModels(Double newEdgePerformance, Double oldFogPerformance, Path edgeModelPath) {
        fogServiceUtils.logInfo("In the aggregation function method");
        List<String> command = modelFileHandler.prepareGeneticAggregationCommand(newEdgePerformance, oldFogPerformance, edgeModelPath);
        modelFileHandler.runScript(command, "genetic");
        concurrencyManager.setAtLeastOneAggregation(Boolean.TRUE);
    }

    /**
     * Provides the current fog model as a downloadable resource.
     * <p>
     * If the fog model file exists, it returns the file as a downloadable resource. Otherwise,
     * it returns an error response.
     * </p>
     *
     * @return a {@link ResponseEntity} containing the fog model file or an error message
     */
    public ResponseEntity<?> requestFogModel() {
        try {
            Path fogModelPath = modelFileHandler.getFogModelPath();
            if (Files.exists(fogModelPath)) {
                InputStreamResource resource = new InputStreamResource(Files.newInputStream(fogModelPath));
                HttpHeaders headers = new HttpHeaders();
                headers.setContentType(MediaType.APPLICATION_OCTET_STREAM);
                headers.setContentDispositionFormData("attachment", pathManager.getFogModelPath());
                return ResponseEntity.ok()
                        .headers(headers)
                        .contentLength(Files.size(fogModelPath))
                        .body(resource);
            } else {
                fogServiceUtils.logError("Fog model file not found at: " + fogModelPath);
                return ResponseEntity.badRequest().body("Fog model file not found.");
            }
        } catch (IOException e) {
            fogServiceUtils.logError("An exception occurred while sending the fog model: " + e.getMessage());
            return ResponseEntity.internalServerError().body("An error occurred while processing the request.");
        }
    }

    /**
     * Starts monitoring the cooling schedule at fixed intervals.
     * <p>
     * This method schedules the {@link MonitorCoolingSchedule} to run at fixed intervals of 1 second.
     * </p>
     */
    public void startMonitoringCoolingSchedule() {
        fogServiceUtils.logInfo("Starting to monitor cooling schedule...");
        concurrencyManager.getScheduler().scheduleAtFixedRate(monitorCoolingSchedule, 0, 1,
                TimeUnit.SECONDS);
    }

    /**
     * Notifies edges about training parameters using the top genetic algorithm parameters.
     *
     * @param trainingEdges the list of edges to notify
     */
    private void notifyEdgesAboutTrainingParameters(List<EdgeEntity> trainingEdges) {
        List<int[]> topParametersList = geneticEngine.getTop3Individuals();

        if (topParametersList.isEmpty()) {
            fogServiceUtils.logError("No parameter sets available to send to edges.");
            return;
        }

        // Iterate over edges and assign parameters
        for (int i = 0; i < trainingEdges.size(); i++) {
            // Use modulus to wrap around if there are more edges than parameter sets
            int[] parameters = topParametersList.get(i % topParametersList.size());

            // Create the body with the current parameter set
            MultiValueMap<String, String> body = createTrainingParametersBody(parameters);
            HttpEntity<MultiValueMap<String, String>> requestEntity = new HttpEntity<>(body,
                    fogServiceUtils.getJsonHeaders());

            EdgeEntity edge = trainingEdges.get(i);
            String endpoint = edge.getEndpoints().get("params");

            fogServiceUtils.logInfo("Sending request to endpoint: " + endpoint + " with body " + body);
            fogServiceUtils.sendPostRequest(endpoint, requestEntity);
        }
    }

    /**
     * Creates a request body containing the training parameters.
     *
     * @param parameters an array of parameter values (e.g., learning rate, batch size, epochs, patience, fine-tune)
     * @return a {@link MultiValueMap} containing the training parameters
     */
    private MultiValueMap<String, String> createTrainingParametersBody(int[] parameters) {
        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        double learningRate = parameters[0] / 10000.0;
        body.add("learning_rate", String.valueOf(learningRate));
        body.add("batch_size", String.valueOf(parameters[1]));
        body.add("epochs", String.valueOf(parameters[2]));
        body.add("patience", String.valueOf(parameters[3]));
        body.add("fine_tune", String.valueOf(parameters[4]));
        return body;
    }

    /**
     * Retrieves the list of elapsed times for each fog iteration.
     *
     * @return a {@link ResponseEntity} containing the list of elapsed times
     */
    public ResponseEntity<?> requestElapsedTimeList() {
        return ResponseEntity.ok(concurrencyManager.getElapsedTimeOneFogIterationList());
    }

    /**
     * Retrieves the list of incoming traffic data over iterations.
     * <p>
     * Loads traffic data from JSON files before returning the results.
     * </p>
     *
     * @return a {@link ResponseEntity} containing the list of incoming traffic data
     */
    public ResponseEntity<?> requestIncomingFogTraffic() {
        fogTraffic.loadTrafficFromJsonFile();
        return ResponseEntity.ok(fogTraffic.getIncomingTrafficOverIterations());
    }

    /**
     * Retrieves the list of outgoing traffic data over iterations.
     * <p>
     * Loads traffic data from JSON files before returning the results.
     * </p>
     *
     * @return a {@link ResponseEntity} containing the list of outgoing traffic data
     */
    public ResponseEntity<?> requestOutgoingFogTraffic() {
        fogTraffic.loadTrafficFromJsonFile();
        System.out.println("outgoing requested traffic: " + fogTraffic.getOutgoingTrafficOverIterations());
        return ResponseEntity.ok(fogTraffic.getOutgoingTrafficOverIterations());
    }

    /**
     * Loads the system state for the genetic engine json's and notifies edges about the fog.
     *
     * @return a {@link ResponseEntity} confirming the system state was loaded successfully
     */
    public ResponseEntity<?> loadSystemState() {
        geneticEngine.loadState();
        fogServiceUtils.notifyEdgeAboutFog();
        return ResponseEntity.ok("The genetic system state loaded successfully.");
    }
}
