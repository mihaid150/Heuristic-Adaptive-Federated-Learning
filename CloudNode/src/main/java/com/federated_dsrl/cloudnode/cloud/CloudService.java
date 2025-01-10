package com.federated_dsrl.cloudnode.cloud;

import com.federated_dsrl.cloudnode.config.AggregationType;
import com.federated_dsrl.cloudnode.config.DeviceManager;
import com.federated_dsrl.cloudnode.config.GeneticEvaluationStrategy;
import com.federated_dsrl.cloudnode.config.PathManager;
import com.federated_dsrl.cloudnode.entity.CloudTraffic;
import com.federated_dsrl.cloudnode.tools.ConcurrencyManager;
import com.federated_dsrl.cloudnode.tools.CloudCoolingSchedule;
import com.federated_dsrl.cloudnode.tools.MonitorReceivedFogModels;
import com.federated_dsrl.cloudnode.tools.StatisticsGenerator;
import com.federated_dsrl.cloudnode.utils.CloudServiceUtils;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import lombok.RequiredArgsConstructor;

import org.apache.commons.lang3.time.StopWatch;
import org.springframework.core.io.FileSystemResource;
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
import java.util.*;
import java.util.concurrent.*;

/**
 * Service class responsible for managing the cloud node's operations, including
 * initializing global models, processing received fog models, and generating various statistics.
 */
@Service
@RequiredArgsConstructor
public class CloudService {
    private final CloudServiceUtils cloudServiceUtils;
    private final PathManager pathManager;
    private final DeviceManager deviceManager;
    private final CloudCoolingSchedule cloudCoolingSchedule;
    private final ConcurrencyManager concurrencyManager;
    private final CloudTraffic cloudTraffic;
    private final MonitorReceivedFogModels monitorReceivedFogModels;
    private final StatisticsGenerator statisticsGenerator;
    private final RestTemplate restTemplate = new RestTemplate();
    private final HttpHeaders formHeaders = createFormHeaders();
    private final StopWatch stopWatch;

    /**
     * Initializes the global process for cloud model creation and monitoring.
     *
     * @param isCacheActive            Indicates whether caching is active.
     * @param geneticEvaluationStrategyArg Genetic evaluation strategy to be used.
     * @param modelType                The type of model to be initialized.
     * @return ResponseEntity indicating the status of the operation.
     * @throws IOException              If an I/O error occurs.
     * @throws InterruptedException     If the process is interrupted.
     */
    public ResponseEntity<?> initializeGlobalProcess(Boolean isCacheActive, String geneticEvaluationStrategyArg,
                                                     String modelType) throws IOException, InterruptedException {

        // parse the genetic evaluation strategy from argument
        GeneticEvaluationStrategy geneticEvaluationStrategy = findGeneticEvaluationStrategy(geneticEvaluationStrategyArg);

        // reset and start the stopWatch for global elapsed time
        stopWatch.reset();
        stopWatch.start();

        // if there were fog models on the device left, delete them to not get in conflict with them
        cloudServiceUtils.deleteFogModelFiles();

        // start the creation of global model
        System.out.println("Initiating the creation stage of global model...");
        ResponseEntity<?> response = executeInitialGlobalProcess(isCacheActive, geneticEvaluationStrategy, modelType);
        System.out.println("Initialization complete, starting to monitor received fog models...");

        // start the listener thread for receiving fog model
        startMonitoringReceivedFogModels();

        return response;
    }

    /**
     * Executes the initial global process, including clearing caches, notifying fog nodes,
     * and broadcasting the dummy model.
     *
     * @param isCacheActive            Indicates whether caching is active.
     * @param geneticEvaluationStrategy The genetic evaluation strategy to be applied.
     * @param modelType                The type of model to be initialized.
     * @return ResponseEntity indicating the status of the operation.
     * @throws IOException              If an I/O error occurs.
     * @throws InterruptedException     If the process is interrupted.
     */
    public ResponseEntity<?> executeInitialGlobalProcess(Boolean isCacheActive, GeneticEvaluationStrategy geneticEvaluationStrategy,
                                                         String modelType)
            throws IOException, InterruptedException {

        // initial settings from cloud utils
        cloudServiceUtils.deleteCacheJsonFolderContent();
        cloudServiceUtils.deleteResultFile();
        cloudServiceUtils.notifyFogAboutCloud(restTemplate);
        cloudServiceUtils.associateEdgesToFogProcess(restTemplate);

        // clear previous saved traffic from the storage
        cloudTraffic.clearTraffic();

        // initiates the json files for monitoring the incoming/outgoing traffic
        cloudTraffic.saveTrafficToJsonFile();

        // resets all the times for measuring elapsed time
        concurrencyManager.clearElapsedTimeOverIterations();

        // clear the list which holds the already dates in which we aggregated data
        concurrencyManager.clearAggregatedDate();

        // initiates the json file for storing the aggregated dates and elapsed time
        concurrencyManager.saveCacheToJsonFile();

        // TODO: replicate also for favg case this notification
        // inform edges what types of model architecture to use based on cloud option(modelType)
        notifyEdgesAboutDummyModelViaFog(modelType);

        // cache locally the model type
        monitorReceivedFogModels.setModelType(modelType);

        // creates the initial random (dummy) model
        File dummyModel = monitorReceivedFogModels.createDummyModel(modelType);

        // broadcast the dummy model to fogs which will forward it to children nodes
        initialBroadcastToFogs(dummyModel, isCacheActive, geneticEvaluationStrategy);
        return ResponseEntity.ok("Initialization and transmission of cloud models successfully terminated!");
    }

    /**
     * Creates HTTP headers for file transmission using multipart form data.
     *
     * @return HttpHeaders configured for multipart form data communication.
     */
    private HttpHeaders createFormHeaders() {
        // create a global communication header for  file transmission
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        return headers;
    }

    /**
     * Notifies edges about the dummy model type via fog nodes.
     *
     * @param modelType The type of dummy model to be used.
     */
    private void notifyEdgesAboutDummyModelViaFog(String modelType) {

        // notification header setting
        HttpHeaders headers = new org.springframework.http.HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);

        // for each fog, send a notification with model type
        for(Map.Entry<String, String> entry : deviceManager.getFogsMap().entrySet()) {
            MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
            body.add("model_type", modelType);
            HttpEntity<MultiValueMap<String, String>> requestEntity = new HttpEntity<>(body, headers);
            ResponseEntity<String> response = restTemplate.postForEntity("http://192.168.2." + entry.getKey()
                    + ":8080/fog/edge-model-type", requestEntity, String.class);
            if (response.getStatusCode().is2xxSuccessful()) {
                System.out.println("Successfully notified fog " + entry.getValue() + " about edge model type.");
            } else {
                System.err.println("Unsuccessfully notified fog " + entry.getValue() + " about edge model type."
                        + ". " + "Status code: " + response.getStatusCode());
            }
        }
    }

    /**
     * Broadcasts the initial dummy model to fog nodes for further transmission to edge nodes.
     *
     * @param dummyModel               The dummy model file to be broadcasted.
     * @param isCacheActive            Indicates whether caching is active.
     * @param geneticEvaluationStrategy The genetic evaluation strategy to be applied.
     */
    private void initialBroadcastToFogs(File dummyModel, Boolean isCacheActive, GeneticEvaluationStrategy
            geneticEvaluationStrategy) {

        // start and end date for initial training chosen based on dataset
        List<String> dates = List.of("2017-04-20", "2018-04-20");
        if (!dummyModel.exists()) {
            System.out.println("Dummy model does not exists.");
        } else {
            FileSystemResource resource = new FileSystemResource(dummyModel);

            // start the cooling schedule of the temperature in the cloud node
            cloudCoolingSchedule.startCloudCoolingScheduleThread();

            // call the broadcast function with arguments the dummy model and the rest of training configuration
            cloudServiceUtils.broadcast(resource, dates, restTemplate, formHeaders, concurrencyManager, isCacheActive,
                    true, AggregationType.GENETIC, geneticEvaluationStrategy);
        }
    }

    /**
     * Performs the daily federation process, broadcasting the global model and managing the training round.
     *
     * @param date                     The date for which the federation is being executed.
     * @param isCacheActive            Indicates whether caching is active.
     * @param geneticEvaluationStrategyArg The genetic evaluation strategy to be used.
     */
    public void dailyFederation(String date, Boolean isCacheActive, String geneticEvaluationStrategyArg) {

        // if cache state option active, load the previous saved state
        if (isCacheActive) {
            loadSystemState();
        }

        // extract from the request argument the genetic strategy used in this training round
        GeneticEvaluationStrategy geneticEvaluationStrategy = findGeneticEvaluationStrategy(geneticEvaluationStrategyArg);

        // reset the stopwatch for round elapsed time measuring
        stopWatch.reset();
        stopWatch.start();

        // load the previous registered traffic from json file and also elapsed time and aggregated dates
        cloudTraffic.loadTrafficFromJsonFile();
        concurrencyManager.loadCacheFromJsonFile();

        // load the previous model file
        File globalModelFile = new File(pathManager.getGlobalModelPath());
        if (globalModelFile.exists()) {
            FileSystemResource resource = new FileSystemResource(globalModelFile);

            // start this round cooling schedule of the temperature
            cloudCoolingSchedule.startCloudCoolingScheduleThread();

            // broadcast the global model to the fogs with this round training configuration
            cloudServiceUtils.broadcast(resource, List.of(date), restTemplate, formHeaders, concurrencyManager,
                    isCacheActive, false, AggregationType.GENETIC, geneticEvaluationStrategy);
        } else {
            System.out.println("Global model file does not exists.");
        }
    }

    /**
     * Processes the result received from a fog node, including saving the model and updating results.
     *
     * @param result      The result data received from the fog node.
     * @param currentDate The current date of the operation.
     * @param fogName     The name of the fog node.
     * @param file        The file containing the fog model.
     * @throws IOException If an I/O error occurs while processing the file.
     */
    public void processReceivedResultFromFog(String result, String currentDate, String fogName, MultipartFile file)
            throws IOException {

        // ensure that only one result is processed at time so use a lock to block the access
        concurrencyManager.getReceivingLock().lock();
        System.out.println("Processing the received result from fog node " + fogName);

        try {
            File directory = new File(pathManager.getCloudDirectoryPath());
            if (!directory.exists() && !directory.mkdir()) {
                throw new IOException("Failed to create directory: " + directory.getAbsolutePath());
            }

            // get the model file name and check if it already exists(reduce repeated transmission effect)
            String fileName = pathManager.getFogModelFileName(fogName);
            File modelFile = new File(directory, fileName);

            if (modelFile.exists()) {
                if (concurrencyManager.getAlreadyAggregated()) {
                    System.out.println("File already exists and already aggregated: " + modelFile.getAbsolutePath() +
                            ". Skipping save and update.");
                    return; // exit the method as the file already exists and aggregation is done
                } else {
                    System.out.println("File already exists but not aggregated yet: " + modelFile.getAbsolutePath() +
                            ". Proceeding to aggregation.");
                }
            } else {
                // save the file as it doesn't exist
                saveFile(file, directory, fileName);
            }

            // TODO check whether is ok to remove this line
            // Save the file as it doesn't exist
            saveFile(file, directory, fileName);

            // append the new results in the json file for results if the received result is not null
            if (result != null) {
                System.out.println("Updating for " + fogName + " with " + result);
                updateResultsJson(result, currentDate, fogName, directory);
            }
        } finally {
            // release the lock after the resources were handled
            concurrencyManager.getReceivingLock().unlock();
        }
    }

    /**
     * Starts monitoring the incoming fog models through a scheduled task.
     */
    public void startMonitoringReceivedFogModels() {
        // thread scheduler executing class MonitorReceivedFogModels (Runnable) for listening to incoming fog models
        System.out.println("Starting to monitor cooling schedule...");
        concurrencyManager.getScheduler().scheduleAtFixedRate(monitorReceivedFogModels, 0, 1,
                TimeUnit.SECONDS);
    }

    /**
     * Saves a file received from a fog node.
     *
     * @param file        The file to be saved.
     * @param directory   The directory where the file will be saved.
     * @param fileName    The name of the file.
     * @throws IOException If an I/O error occurs while saving the file.
     */
    private void saveFile(MultipartFile file, File directory, String fileName) throws IOException {
        // TODO check if can be moved to a utility class
        // utility function to save the received file
        File modelFile = new File(directory, fileName);
        file.transferTo(modelFile);
        System.out.println("Model file saved to: " + modelFile.getAbsolutePath());
        // to avoid many printings
        concurrencyManager.getFoundPrinterCounter().set(0);
    }

    /**
     * Updates the results JSON file with the received data from a fog node.
     *
     * @param result      The result data.
     * @param currentDate The current date of the operation.
     * @param fogName     The name of the fog node.
     * @param directory   The directory where the results file is stored.
     * @throws IOException If an I/O error occurs while updating the file.
     */
    private void updateResultsJson(String result, String currentDate, String fogName, File directory)
            throws IOException {
        // TODO check if can be moved to a utility class
        // utility function to append the received results in the json file
        File resultsFile = new File(directory, "results.json");
        Map<String, Map<String, String>> resultsMap = readResultsFromJson(resultsFile);

        Map<String, String> fogData = new HashMap<>();
        fogData.put("result", result);
        fogData.put("currentDate", currentDate);
        resultsMap.put(fogName, fogData);

        writeResultsToJson(resultsMap, resultsFile);
    }

    /**
     * Reads the existing results from a JSON file.
     *
     * @param resultsFile The JSON file containing the results.
     * @return A map of fog names to their results.
     * @throws IOException If an I/O error occurs while reading the file.
     */
    private Map<String, Map<String, String>> readResultsFromJson(File resultsFile) throws IOException {
        // TODO check if can be moved to a utility class
        // utility function to read already written results in the json file
        if (resultsFile.exists()) {
            try (FileReader reader = new FileReader(resultsFile)) {
                return new Gson().fromJson(reader, new TypeToken<Map<String, Map<String, String>>>() {
                }.getType());
            }
        } else {
            return new HashMap<>();
        }
    }

    /**
     * Writes the results to a JSON file.
     *
     * @param resultsMap  The map of results to be written.
     * @param resultsFile The JSON file where results will be saved.
     * @throws IOException If an I/O error occurs while writing to the file.
     */
    private void writeResultsToJson(Map<String, Map<String, String>> resultsMap, File resultsFile) throws IOException {
        // TODO check if can be moved to a utility class
        // utility function to write results in the json file
        try (FileWriter writer = new FileWriter(resultsFile)) {
            new GsonBuilder().setPrettyPrinting().create().toJson(resultsMap, writer);
        }
    }

    /**
     * Retrieves the current cooling temperature.
     *
     * @return ResponseEntity containing the cooling temperature.
     */
    public ResponseEntity<?> getCoolingTemperature() {
        // get method to return as response when requested the cooling cooler temperature
        if (cloudCoolingSchedule.getIsCoolingOperational()) {
            return ResponseEntity.ok(cloudCoolingSchedule.getTemperature());
        } else {
            return ResponseEntity.ok(0.0);
        }
    }

    /**
     * Creates a chart visualizing the elapsed time for global operations.
     *
     * @return ResponseEntity indicating the status of the operation.
     * @throws IOException          If an I/O error occurs.
     * @throws InterruptedException If the process is interrupted.
     */
    public ResponseEntity<?> createElapsedTimeChart() throws IOException, InterruptedException {
        // initiate the creation of elapsed global time chart
        return statisticsGenerator.createElapsedTimeChart();
    }

    /**
     * Creates a chart visualizing the elapsed time for each fog node.
     *
     * @return ResponseEntity indicating the status of the operation.
     */
    public ResponseEntity<?> createElapsedTimeChartFogLayer() {
        // initiate the creation of elapsed time chart for each fog
        return statisticsGenerator.createElapsedTimeChartsFogLayer(restTemplate);
    }

    /**
     * Creates a chart visualizing the traffic data across nodes.
     *
     * @return ResponseEntity indicating the status of the operation.
     */
    public ResponseEntity<?> createTrafficChart() {
        // initiate the creation of overall traffic chart
        return statisticsGenerator.createTrafficChart(restTemplate);
    }

    /**
     * Creates a chart visualizing the performance of the system.
     *
     * @return ResponseEntity indicating the status of the operation.
     */
    public ResponseEntity<?> createPerformanceChart() {
        // initiate the creation of performance chart
        return statisticsGenerator.createPerformanceChart(restTemplate);
    }

    /**
     * Loads the system state by notifying fogs and loading previously saved configurations.
     *
     * @return ResponseEntity indicating the status of the operation.
     */
    public ResponseEntity<?> loadSystemState() {
        // retransmit the network configuration to children nodes
        cloudServiceUtils.notifyFogAboutCloud(restTemplate);
        cloudServiceUtils.associateEdgesToFogProcess(restTemplate);

        // make a request to each fog to load previous training state which were saved locally
        deviceManager.getFogsMap().forEach((fogHost, fogName) -> {
            String url = "http://192.168.2." + fogHost + ":8080/fog/load-system-state";
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body,
                    cloudServiceUtils.getJsonHeaders());
            ResponseEntity<String> response = restTemplate.postForEntity(url, requestEntity, String.class);
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                System.out.println("State loaded for " + fogName);
            }
        });
        return ResponseEntity.ok("Loaded fog states.");
    }

    /**
     * Parses the genetic evaluation strategy from a string argument.
     *
     * @param geneticEvaluationStrategyArg The string representation of the genetic evaluation strategy.
     * @return The corresponding GeneticEvaluationStrategy enum value.
     */
    private GeneticEvaluationStrategy findGeneticEvaluationStrategy(String geneticEvaluationStrategyArg) {
        geneticEvaluationStrategyArg = geneticEvaluationStrategyArg.replace("\"", "");

        // default genetic strategy if the provided one cannot be parsed successfully
        GeneticEvaluationStrategy geneticEvaluationStrategy = GeneticEvaluationStrategy.UPDATE_BOTTOM_INDIVIDUALS;
        try {
            geneticEvaluationStrategy = GeneticEvaluationStrategy.valueOf(geneticEvaluationStrategyArg.toUpperCase());
        } catch(IllegalArgumentException e) {
            System.out.println("Invalid genetic evaluation strategy: " + e.getMessage());
        }
        return geneticEvaluationStrategy;
    }
}
