package com.federated_dsrl.cloudnode.tools;

import com.federated_dsrl.cloudnode.entity.CloudTraffic;
import com.federated_dsrl.cloudnode.handlers.AggregationWebSocketHandler;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.reflect.TypeToken;
import com.federated_dsrl.cloudnode.config.DeviceManager;
import com.federated_dsrl.cloudnode.config.PathManager;
import com.federated_dsrl.cloudnode.utils.CloudServiceUtils;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.apache.commons.lang3.time.StopWatch;
import org.springframework.stereotype.Component;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

/**
 * Monitors and processes received fog model files for aggregation at the cloud node.
 * <p>
 * This class implements a listener mechanism to detect when all fog models have been received and triggers
 * the appropriate aggregation process (e.g., FedAvg or heuristic-based aggregation). It handles file reading,
 * aggregation, and cleanup after the aggregation process.
 * </p>
 */
@Component
@RequiredArgsConstructor
public class MonitorReceivedFogModels implements Runnable {

    /**
     * Manages concurrency utilities such as locks, schedules, and counters.
     */
    private final ConcurrencyManager concurrencyManager;

    /**
     * Provides paths to various files and directories used in the cloud node.
     */
    private final PathManager pathManager;

    /**
     * Manages devices (fogs and edges) associated with the cloud node.
     */
    private final DeviceManager deviceManager;

    /**
     * Stopwatch used to measure elapsed time during aggregation and monitoring processes.
     */
    private final StopWatch stopWatch;

    /**
     * Manages cooling schedules for the cloud node.
     */
    private final CloudCoolingSchedule cloudCoolingSchedule;

    /**
     * Tracks traffic statistics for incoming and outgoing requests.
     */
    private final CloudTraffic cloudTraffic;

    /**
     * Utility class providing common cloud-related functionalities.
     */
    private final CloudServiceUtils cloudServiceUtils;

    /**
     * WebSocket handler for notifying clients about aggregation completion.
     */
    private final AggregationWebSocketHandler aggregationWebSocketHandler;

    /**
     * Specifies the type of model being processed.
     */
    @Setter
    private String modelType;

    /**
     * Monitors the directory for received fog model files.
     * <p>
     * When all expected fog model files are received, it initiates the aggregation process and performs
     * necessary post-aggregation tasks such as clearing files and storing results.
     * </p>
     */
    @Override
    public void run() {
        // listener method to wait for fog aggregated models
        Path cloudDirectoryPath = Paths.get(pathManager.getCloudDirectoryPath());

        if (!Files.exists(cloudDirectoryPath)) {
            if (concurrencyManager.getPrintingCounter().get() < concurrencyManager.getAllowedPrintingTimes()) {
                // for max printing times purposes
                System.out.println("Cloud directory path does not exist yet.");
                concurrencyManager.getPrintingCounter().getAndIncrement();
            }
            return;
        }
        concurrencyManager.getPrintingCounter().set(0);

        try (Stream<Path> paths = Files.list(cloudDirectoryPath)) {

            // trying to count how many fog model files have arrived
            long count = paths
                    .filter(path -> path.getFileName().toString().startsWith("fog_model_"))
                    .count();

            if (concurrencyManager.getFoundPrinterCounter().get() < concurrencyManager.getAllowedPrintingTimes()) {
                // for not to many times printing reason
                System.out.println("Found " + count + " fog model files.");
                concurrencyManager.getFoundPrinterCounter().getAndIncrement();

                if (count == deviceManager.getFogsMap().size()) {
                    System.out.println("All fog model files received. Initiating aggregation process.");
                }
            }

            // have received all trained fog models in this round
            if (count == deviceManager.getFogsMap().size()) {

                // use a lock to block access to resources for unintended requests
                concurrencyManager.getReceivingLock().lock();

                // no need to cool down the temperature this round
                cloudCoolingSchedule.stopCloudCoolingScheduleThread();

                // printing purpose
                concurrencyManager.getWaiterCounter().set(0);

                // get the results json file
                File directory = new File(pathManager.getCloudDirectoryPath());
                File resultsFile = new File(directory, "results.json");

                // when a results file exits means that we are in the heuristic training strategy branch
                if (resultsFile.exists()) {
                    Map<String, Map<String, String>> resultsMap;

                    // read the results.json file into resultsMap
                    try (FileReader reader = new FileReader(resultsFile)) {
                        resultsMap = new Gson().fromJson(reader, new TypeToken<Map<String, Map<String, String>>>() {
                        }.getType());
                    } catch (IOException e) {
                        System.err.println("Error reading results.json file: " + e.getMessage());
                        return;
                    }

                    // test if all the results are from the same date
                    if (allCurrentDatesMatch(resultsMap)) {

                        // check if is the initial aggregation
                        if (concurrencyManager.getLastAggregatedDate() == null || !concurrencyManager
                                .getLastAggregatedDate().equals(getAggregatedDate(resultsMap))) {
                            concurrencyManager.addNewAggregatedDate(getAggregatedDate(resultsMap));

                            // parse the lambda values from the results file
                            Map<String, Double> fogResults = new HashMap<>();
                            for (Map.Entry<String, Map<String, String>> entry : resultsMap.entrySet()) {
                                String jsonString = entry.getValue().get("result");
                                JsonObject jsonObject = JsonParser.parseString(jsonString).getAsJsonObject();
                                Double lambdaPrev = jsonObject.get("lambda_prev").getAsDouble();
                                fogResults.put(entry.getKey(), lambdaPrev);
                            }
                            try {
                                // if all ok so far, call the aggregation method on the received models and results
                                aggregateModels(Objects.requireNonNull(getGlobalModelFile()),
                                        getFogModels(directory), fogResults);
                                System.out.println("Aggregation process completed successfully.");
                                postAggregationUtils();
                            } catch (Exception e) {
                                System.err.println("Error during aggregation process: " + e.getMessage());
                            }
                        }
                    }
                } else {
                    // this is the case for FedAvg(Favg) training and aggregation process

                    if (concurrencyManager.getLastAggregatedDate().equals("2013-07-09") &&
                            !concurrencyManager.getAlreadyAggregated()) {
                        // initial aggregation
                        concurrencyManager.setAlreadyAggregated(Boolean.TRUE);
                        aggregateModelsFavg(Objects.requireNonNull(getGlobalModelFile()), getFogModels(directory));
                        postAggregationUtils();
                    } else {
                        if (!concurrencyManager.getAlreadyAggregated()) {

                            // all 3 fog models present but no results file -> simple favg with no need to cache results
                            concurrencyManager.setAlreadyAggregated(Boolean.TRUE);
                            aggregateModelsFavg(Objects.requireNonNull(getGlobalModelFile()), getFogModels(directory));
                            postAggregationUtils();
                        }
                    }
                }
                concurrencyManager.getReceivingLock().unlock();
            } else {
                if (concurrencyManager.getWaiterCounter().get() < concurrencyManager.getAllowedPrintingTimes()) {
                    System.out.println("Waiting for more fog model files. Current count: " + count);
                    concurrencyManager.getWaiterCounter().getAndIncrement();
                }
            }
        } catch (IOException e) {
            System.err.println("Error listing files in cloud directory: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Unexpected error in MonitorReceivedFogModels: " + e.getMessage());
        }
    }

    /**
     * Checks whether all results in the provided map correspond to the same date.
     *
     * @param resultMap A map containing fog results with their associated dates.
     * @return {@code true} if all results have the same date, otherwise {@code false}.
     */
    private Boolean allCurrentDatesMatch(Map<String, Map<String, String>> resultMap) {
        // check if all the results are from the same day
        List<String> dates = new ArrayList<>();
        resultMap.values().forEach(entry -> dates.add(entry.get("currentDate")));
        return dates.stream().allMatch(date -> date.equals(dates.get(0)));
    }

    /**
     * Retrieves the last aggregation date from the provided results map.
     *
     * @param resultsMap A map containing fog results with their associated dates.
     * @return The last aggregation date as a string.
     */
    private String getAggregatedDate(Map<String, Map<String, String>> resultsMap) {
        // get the last aggregation date
        List<String> dates = new ArrayList<>();
        resultsMap.values().forEach(entry -> dates.add(entry.get("currentDate")));
        return dates.get(0).replace("]", "");
    }

    /**
     * Creates a dummy model file based on the specified model type.
     *
     * @param modelType The type of model to create (e.g., "LSTM").
     * @return The created dummy model file.
     * @throws IOException If an error occurs during file creation.
     * @throws InterruptedException If the process is interrupted during execution.
     */
    public File createDummyModel(String modelType) throws IOException, InterruptedException {

        // creation a dummy(random) initial architecture model
        // load the executable for model creation and set up the arguments
        File scriptFile = loadPythonScript(pathManager.getCreateInitLSTMModelScript());
        List<String> command = new ArrayList<>();
        command.add(pathManager.getPython3ExecutablePath());
        command.add(scriptFile.getAbsolutePath());
        command.add(modelType);
        Process process = runProcess(command);

        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        StringBuilder output = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            output.append(line).append("\n");
        }
        // debug purpose
        System.out.println("Python create_init_lstm_model (global) script output: " + output);

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            StringBuilder errorOutput = new StringBuilder();
            while ((line = errorReader.readLine()) != null) {
                errorOutput.append(line).append("\n");
            }
            System.err.println("Error output from python create_init_lstm_model (global) script: " + errorOutput);
            throw new RuntimeException("Failed to execute python script create_init_lstm_model (global) " +
                    "with exit code: " + exitCode);
        }

        String modelPath = "/app/models/cloud/global_model.keras";
        File modelFile = new File(modelPath);

        if (!modelFile.exists()) {
            throw new FileNotFoundException("Model file was not found at the expected location: " + modelPath);
        }

        return modelFile;
    }

    /**
     * Aggregates fog models based on heuristic strategies using specified lambda values.
     *
     * @param globalModelFile The global model file.
     * @param fogModelFiles A map of fog model files.
     * @param lambda_prevs A map of lambda values associated with each fog model.
     */
    private void aggregateModels(File globalModelFile, Map<String, File> fogModelFiles, Map<String, Double> lambda_prevs) {
        // aggregation method for the fog models in the heuristic branch
        // loading the executable file for aggregation and set up the arguments
        System.out.println("Running the aggregation in the cloud...");
        File scriptFile = new File(pathManager.getAggregatedModelPath());
        List<String> command = new ArrayList<>();
        command.add(pathManager.getPython3ExecutablePath());
        command.add(scriptFile.getAbsolutePath());
        command.add(globalModelFile.getAbsolutePath());

        // for each fog entry, add its fog model file and the associated lambda to arguments list
        for (Map.Entry<String, File> entry : fogModelFiles.entrySet()) {
            File fogFile = entry.getValue();
            Double lambdaPrev = lambda_prevs.get(entry.getKey());

            command.add(fogFile.getAbsolutePath());
            command.add(String.valueOf(lambdaPrev));
        }

        // create the process which will run the aggregation script
        try {
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.redirectErrorStream(true);
            Process process = processBuilder.start();

            StringBuilder output = provideProcessOutput(process);
            System.out.println("Aggregation script output from cloud node: " + output);

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                throw new RuntimeException("Failed to execute aggregation in the cloud script with exit code: " + exitCode);
            }
        } catch (IOException | InterruptedException e) {
            System.out.println("Exception generated from executing aggregation method in the cloud.");
        }
    }

    /**
     * Loads a Python script from the specified path.
     *
     * @param scriptPath The path to the Python script.
     * @return The {@link File} object representing the Python script.
     * @throws IOException If the script file does not exist or cannot be accessed.
     */
    private File loadPythonScript(String scriptPath) throws IOException {
        File scriptFile = new File(scriptPath);

        // Check if the script file exists at the given path
        if (!scriptFile.exists()) {
            throw new FileNotFoundException("Script not found at path: " + scriptPath);
        }

        return scriptFile;
    }

    /**
     * Retrieves the global model file, creating a new one if it does not exist.
     *
     * @return The global model file.
     */
    private File getGlobalModelFile() {
        // load the global file if exists and if not creates a new one
        Path path = Paths.get(pathManager.getGlobalModelPath());
        File globalModelFile;
        if (!Files.exists(path)) {
            try {
                globalModelFile = createDummyModel(modelType);
                System.out.println("Global model file created: " + globalModelFile.getAbsolutePath());
            } catch (IOException | InterruptedException e) {
                System.err.println("Error creating dummy global model file: " + e.getMessage());
                return null;
            }
        } else {
            globalModelFile = new File(pathManager.getGlobalModelPath());
            System.out.println("Global model file exists: " + globalModelFile.getAbsolutePath());
        }
        return globalModelFile;
    }

    /**
     * Retrieves fog model files from the specified directory.
     *
     * @param directory The directory containing fog model files.
     * @return A map of fog model names to their corresponding files.
     */
    private Map<String, File> getFogModels(File directory) {
        // gets the locally saved fog model files in a map
        Map<String, File> fogModelFiles = new HashMap<>();

        for (String fogName : deviceManager.getFogsMap().values()) {
            File fogModelFile = new File(directory, "fog_model_" + fogName + ".keras");
            if (fogModelFile.exists()) {
                fogModelFiles.put(fogName, fogModelFile);
                System.out.println("Fog model file found for fogName: " + fogName);
            } else {
                System.err.println("Fog model file not found for fogName: " + fogName);
            }
        }
        return fogModelFiles;
    }

    /**
     * Aggregates fog models using the FedAvg strategy.
     *
     * @param globalModelFile The global model file.
     * @param fogModels A map of fog model files.
     * @throws IOException If an error occurs during file handling.
     * @throws InterruptedException If the process is interrupted during execution.
     */
    private void aggregateModelsFavg(File globalModelFile, Map<String, File> fogModels) throws IOException, InterruptedException {
        // aggregation method for the fog models in the favg branch
        // loading the executable file for aggregation and set up the arguments
        System.out.println("Running aggregation in the cloud favg case...");
        File scriptFile = new File(pathManager.getAggregatedFavgModelPath());
        List<String> command = new ArrayList<>();
        command.add(pathManager.getPython3ExecutablePath());
        command.add(scriptFile.getAbsolutePath());
        command.add(globalModelFile.getAbsolutePath());

        // load and add as argument each fog model file
        for (Map.Entry<String, File> entry : fogModels.entrySet()) {
            File fogFile = entry.getValue();
            command.add(fogFile.getAbsolutePath());
        }

        // create the process for running the aggregation executable
        Process process = runProcess(command);
        StringBuilder output = provideProcessOutput(process);
        System.out.println("Aggregation script output from cloud node: " + output);
        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new RuntimeException("Failed to execute aggregation in the cloud script with exit code: " + exitCode);
        }
    }


    /**
     * Executes a process with the specified command.
     *
     * @param command A list of command-line arguments to run the process.
     * @return The {@link Process} instance representing the running process.
     * @throws IOException If an error occurs while starting the process.
     */
    private Process runProcess(List<String> command) throws IOException {
        // util for running a process
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.redirectErrorStream(true);
        return processBuilder.start();
    }

    /**
     * Parses the output of a running process and filters out unwanted logs.
     *
     * @param process The process whose output is to be captured.
     * @return A {@link StringBuilder} containing the filtered output of the process.
     * @throws IOException If an error occurs while reading the process output.
     */
    private StringBuilder provideProcessOutput(Process process) throws IOException {
        // parse the output from a ran process and delete some unwanted logs
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        StringBuilder output = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            if (!line.equals("WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be " +
                    "built. `model.compile_metrics` will be empty until you train or evaluate the model.")) {
                output.append(line).append("\n");
            }
        }
        return output;
    }

    /**
     * Performs post-aggregation tasks such as clearing fog model files, storing traffic statistics, and
     * notifying clients about aggregation completion.
     */
    private void postAggregationUtils() {
        // some util instructions after the aggregation has executed

        // deleted the fog model files as there is no need for them
        cloudServiceUtils.deleteFogModelFiles();

        // stop the watch for elapsed time counting
        stopWatch.stop();

        // process elapsed time
        double elapsedTime = stopWatch.getTime() / 1000.0;
        concurrencyManager.getElapsedTimeOverIterations()
                .put(concurrencyManager.getLastAggregatedDate(), elapsedTime);
        System.out.println("Total elapsed time for the iteration of date " +
                concurrencyManager.getLastAggregatedDate() + " is " + elapsedTime);

        // store the traffic in the list for the current iteration
        cloudTraffic.storeCurrentIterationIncomingTraffic();
        cloudTraffic.storeCurrentIterationOutgoingTraffic();

        // save to json the traffic lists
        cloudTraffic.saveTrafficToJsonFile();

        // save to json the elapsed time and aggregation dates
        concurrencyManager.saveCacheToJsonFile();

        System.out.println("Cloud incoming traffic: " + cloudTraffic.getIncomingTrafficOverIterations());
        System.out.println("Cloud outgoing traffic: " + cloudTraffic.getOutgoingTrafficOverIterations());

        // clear the traffic lists
        cloudTraffic.resetIncomingTraffic();
        cloudTraffic.resetOutgoingTraffic();

        // notify the web client for the end of aggregation
        aggregationWebSocketHandler.notifyAggregationComplete("AggregationCompleted");
    }
}
