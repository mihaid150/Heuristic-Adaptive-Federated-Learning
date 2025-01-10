package org.federated_dsrl.edgenode.edge;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.federated_dsrl.edgenode.config.AggregationType;
import org.federated_dsrl.edgenode.config.DeviceManager;
import org.federated_dsrl.edgenode.config.PathManager;
import org.federated_dsrl.edgenode.config.TrainingParametersManager;
import org.federated_dsrl.edgenode.entity.EdgeTraffic;
import org.federated_dsrl.edgenode.entity.EdgeTrafficManager;
import org.federated_dsrl.edgenode.tools.DataManager;
import org.federated_dsrl.edgenode.utils.EdgeServiceUtils;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.http.HttpHeaders;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Service
@RequiredArgsConstructor
public class EdgeService {
    private final DataManager dataManager;
    private final DeviceManager deviceManager;
    private final PathManager pathManager;
    private final EdgeServiceUtils edgeServiceUtils;
    private final TrainingParametersManager trainingParametersManager;
    private final EdgeTrafficManager edgeTrafficManager;

    /**
     * Sets the training parameters for the edge service.
     *
     * @param learningRate The learning rate for training.
     * @param batchSize    The batch size for training.
     * @param epochs       The number of epochs for training.
     * @param patience     The early stopping patience value.
     * @param fineTune     The number of fine-tuning layers.
     * @return A response confirming the training parameters were set successfully.
     */
    public ResponseEntity<?> setTrainingParameters(String learningRate, String batchSize, String epochs, String patience,
                                                   String fineTune) {
        trainingParametersManager.setLearningRate(Double.parseDouble(learningRate));
        trainingParametersManager.setBatchSize(Integer.parseInt(batchSize));
        trainingParametersManager.setEpochs(Integer.parseInt(epochs));
        trainingParametersManager.setEarlyStoppingPatience(Integer.parseInt(patience));
        trainingParametersManager.setFineTuneLayers(Integer.parseInt(fineTune));
        System.out.println("Training parameters were set!: " + trainingParametersManager);
        return ResponseEntity.ok("Training parameters were set!: " + trainingParametersManager);
    }

    /**
     * Identifies the parent fog node for the edge service.
     *
     * @param host The host address of the parent fog node.
     */
    public void identifyParentFog(String host) {
        deviceManager.updateMapping(host);
        System.out.println("host " + host + "set");
    }

    /**
     * Sets the model type to be used for training.
     *
     * @param modelType The type of the model (e.g., LSTM).
     */
    public void setModelType(String modelType) {
        System.out.println("Model type is: " + modelType);
        trainingParametersManager.setModelType(modelType);
    }

    /**
     * Retrains the fog model with the provided file, dates, and configuration.
     *
     * @param file              The fog model file.
     * @param date              The training date(s).
     * @param lclid             The logical cluster ID.
     * @param aggregationTypeArg The aggregation type used for training.
     * @throws IOException          If an error occurs during file handling.
     * @throws InterruptedException If the thread is interrupted during training.
     */
    public void retrainFogModel(MultipartFile file, List<String> date, String lclid,
                                String aggregationTypeArg)
            throws IOException, InterruptedException {

        aggregationTypeArg = aggregationTypeArg.replace("\"", "");
        AggregationType aggregationType = AggregationType.GENETIC;
        try {
            aggregationType = AggregationType.valueOf(aggregationTypeArg.toUpperCase());
        } catch (IllegalArgumentException e) {
            System.out.println("Invalid aggregation type: " + aggregationTypeArg);
        }

        //edgeTrafficManager.loadTrafficFromJsonFile();
        dataManager.loadPerformanceFromJson();
        String formattedTrainingDate = extractDate(date.get(0));
        System.out.println("Formatted start date: " + formattedTrainingDate + " -> " + lclid);
        String dataJsonPath;
        dataJsonPath = dataManager.provideDailyData(formattedTrainingDate, lclid);

        long timeout = 5 * 60 * 1000;
        long startTime = System.currentTimeMillis();

        while (dataJsonPath == null || !new File(dataJsonPath).exists()) {
            // Check if timeout has been reached
            System.out.println("waiting for the daily data...");
            if (System.currentTimeMillis() - startTime > timeout) {
                throw new IOException("Timeout waiting for the dataJsonPath file to be created");
            }
            // Wait for a short period before checking again
            Thread.sleep(500);
        }

        Path modelsDirectory = Paths.get(pathManager.getModelsDirectory()).toAbsolutePath();
        Path fogModelPath = modelsDirectory.resolve(Objects.requireNonNull(pathManager.getEnhancedModelPath()));
        Files.write(fogModelPath, file.getBytes());
        File fogModel = new File(String.valueOf(fogModelPath));
        System.out.println("edge retrain fog model -> received fog model path: " + fogModelPath);

        Path currentDir = Paths.get("").toAbsolutePath();
        //File scriptFile = extractScript(pathManager.getClientEnvScriptPath());
        File scriptFile = new File(pathManager.getClientEnvScriptPath());

        signalReadinessToFog(lclid, aggregationType);
        waitForFogSignal(lclid, aggregationType);
        List<String> command;

        if (aggregationType.equals(AggregationType.GENETIC)) {
            command = List.of(
                    pathManager.getShortPythonExecutablePath(),
                    scriptFile.getAbsolutePath(),
                    dataJsonPath,
                    fogModel.toString(),
                    formattedTrainingDate,
                    lclid,
                    trainingParametersManager.getLearningRate().toString(),
                    trainingParametersManager.getBatchSize().toString(),
                    trainingParametersManager.getEpochs().toString(),
                    trainingParametersManager.getEarlyStoppingPatience().toString(),
                    trainingParametersManager.getFineTuneLayers().toString(),
                    "not_first"
            );
        } else {
            command = List.of(
                    pathManager.getShortPythonExecutablePath(),
                    scriptFile.getAbsolutePath(),
                    dataJsonPath,
                    fogModel.toString(),
                    formattedTrainingDate,
                    lclid,
                    Double.toString(0.001),
                    Integer.toString(32),
                    Integer.toString(10),
                    Integer.toString(5),
                    Integer.toString(3),
                    "not_first"
            );
        }

        workHelper(command, currentDir, lclid, formattedTrainingDate, aggregationType);
    }

    /**
     * Executes a Python script and parses its output.
     *
     * @param processBuilder The process builder for the script execution.
     * @return A map containing the parsed output from the script.
     * @throws IOException          If an error occurs during script execution or parsing.
     * @throws InterruptedException If the thread is interrupted while waiting for the script to finish.
     */
    private Map<String, Map<String, Double>> runPythonString(ProcessBuilder processBuilder) throws
            IOException, InterruptedException {
        processBuilder.redirectErrorStream(true);
        Process process = processBuilder.start();

        StringWriter outputWriter = new StringWriter();
        StringWriter errorWriter = new StringWriter();

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
             BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line + "\n");
                outputWriter.write(line + "\n");
            }
            while ((line = errorReader.readLine()) != null) {
                System.out.println(line + "\n");
                errorWriter.write(line + "\n");
            }
        }

        int exitCode = process.waitFor();
        System.out.println("Python script client_env exited with code: " + exitCode);

        String output = outputWriter.toString();
        String errorOutput = errorWriter.toString();
        if (!errorOutput.isEmpty()) {
            System.err.println("Python script client_env error output: " + errorOutput);
        }

        // Find and parse the JSON content from the output
        String jsonString = getString(output);

        // Assuming the output is in JSON format and should be converted to Map<String, Map<String, Double>>
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        objectMapper.configure(JsonParser.Feature.ALLOW_NON_NUMERIC_NUMBERS, true); // Allow NaN

        // Deserialize into a nested map structure
        return objectMapper.readValue(jsonString, new TypeReference<>() {
        });
    }

    /**
     * Extracts the JSON string content from the provided script output.
     * <p>
     * The method identifies and validates the JSON output by locating the starting index of the JSON content.
     * </p>
     *
     * @param output The output string generated by the Python script.
     * @return A string containing the JSON content extracted from the script output.
     * @throws IOException If the JSON content is not found or the output is invalid.
     */
    private static String getString(String output) throws IOException {
        int jsonStartIndex = output.indexOf("{");
        if (jsonStartIndex == -1) {
            throw new IOException("Python script client_env did not return valid JSON output.");
        }

        String jsonString = output.substring(jsonStartIndex).trim();

        // Ensure there is content before attempting to parse it
        if (jsonString.isEmpty()) {
            throw new IOException("Python script client_env did not return valid JSON output.");
        }
        return jsonString;
    }

    /**
     * Sends the result and model to the parent fog node.
     *
     * @param fogPerformance  The performance of the fog model.
     * @param edgePerformance The performance of the edge model.
     * @param bestModelPath   The path to the best model file.
     * @param lclid           The logical cluster ID.
     * @param aggregationType The aggregation type used for training.
     */
    private void sendResultToFog(Double fogPerformance, Double edgePerformance, String bestModelPath, String lclid,
                                 AggregationType aggregationType) {
        System.out.println("aggregation: " + aggregationType);
        RestTemplate restTemplate = new RestTemplate();
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        File modelFile = new File(bestModelPath);
        if (!modelFile.exists()) {
            System.err.println("Model file does not exist: " + bestModelPath);
            return;
        }

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("new_edge_performance", edgePerformance);
        body.add("old_fog_performance", fogPerformance);
        body.add("lclid", lclid);
        body.add("edge_model", new FileSystemResource(modelFile));

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        String url = aggregationType.equals(AggregationType.GENETIC) ? deviceManager.getApis().get("receive") :
                deviceManager.getApis().get("receive-favg");

        try {
            ResponseEntity<String> response = restTemplate.postForEntity(url, requestEntity, String.class);
            if (response.getStatusCode().is2xxSuccessful()) {
                System.out.println("Successfully sent result and model to the parent fog.");
            } else {
                System.err.println("Failed to send result and model to the parent fog. Status code: " + response.getStatusCode());
            }
        } catch (Exception e) {
            System.err.println("Exception occurred while sending result and model to the parent fog: " + e.getMessage());
        }
    }

    /**
     * Creates an initial model using a Python script.
     *
     * @return A file reference to the created initial model.
     * @throws IOException          If an error occurs during script execution.
     * @throws InterruptedException If the thread is interrupted while waiting for the script to finish.
     */
    private File create_init_model() throws IOException, InterruptedException {

        File scriptFile = new File(pathManager.getCreateInitLSTMModelFilePath());
        List<String> command = new ArrayList<>();
        command.add(pathManager.getPythonVirtualExecutablePath());
        command.add(scriptFile.getAbsolutePath());
        command.add(trainingParametersManager.getModelType());

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        Process process = processBuilder.start();

        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        StringBuilder output = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            output.append(line).append("\n");
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            StringBuilder errorOutput = new StringBuilder();
            while ((line = errorReader.readLine()) != null) {
                errorOutput.append(line).append("\n");
            }
            System.err.println("Error output from Python script: " + errorOutput);
            throw new RuntimeException("Failed to execute Python script with exit code: " + exitCode);
        }
        System.out.println("edge service create init model output: " + output);
        String modelPath = output.toString().trim().substring(15); // extract the model path correctly
        System.out.println("modelPath: " + modelPath);
        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            throw new IOException("Model file does not exist: " + modelFile.getAbsolutePath());
        }
        return modelFile;
    }

    /**
     * Initializes the edge service with data and configurations.
     *
     * @param startDate         The start date for the initialization.
     * @param endDate           The end date for the initialization.
     * @param lclid             The logical cluster ID.
     * @param aggregationTypeArg The aggregation type used for training.
     * @throws IOException          If an error occurs during file handling.
     * @throws InterruptedException If the thread is interrupted during the initialization process.
     */
    public void initialization(String startDate, String endDate, String lclid, String aggregationTypeArg)
            throws IOException, InterruptedException {

        edgeServiceUtils.deleteCacheJsonFolderContent();
        aggregationTypeArg = aggregationTypeArg.replace("\"", "");
        AggregationType aggregationType = AggregationType.GENETIC;
        try {
            aggregationType = AggregationType.valueOf(aggregationTypeArg.toUpperCase());
        } catch (IllegalArgumentException e) {
            System.out.println("Invalid aggregation type: " + aggregationTypeArg);
        }

        //edgeTrafficManager.loadTrafficFromJsonFile();
        String formattedStartDate = extractDate(startDate);
        String formattedEndDate = extractDate(endDate);

        System.out.println("Formatted start date: " + formattedStartDate + " -> " + "Formatted end date: " +
                formattedEndDate + " -> " + lclid);
        String dataJsonPath;
        dataJsonPath = dataManager.providePeriodData(formattedStartDate, formattedEndDate, lclid);

        // Timeout in milliseconds (5 minutes)
        long timeout = 5 * 60 * 1000;
        long startTime = System.currentTimeMillis();

        // Check if dataJsonPath file exists and wait until it does
        while (dataJsonPath == null || !new File(dataJsonPath).exists()) {
            // Check if timeout has been reached
            System.out.println("waiting for the period data...");
            if (System.currentTimeMillis() - startTime > timeout) {
                throw new IOException("Timeout waiting for the dataJsonPath file to be created");
            }
            // Wait for a short period before checking again
            Thread.sleep(500);
        }

        File modelFile = create_init_model();
        Path currentDir = Paths.get("").toAbsolutePath();

        //File scriptFile = extractScript();
        File scriptFile = new File(pathManager.getClientEnvScriptPath());

        signalReadinessToFog(lclid, aggregationType);
        waitForFogSignal(lclid, aggregationType);
        List<String> command;
        if (aggregationType.equals(AggregationType.GENETIC)) {
            command = List.of(
                    pathManager.getShortPythonExecutablePath(),
                    scriptFile.getAbsolutePath(),
                    dataJsonPath,
                    modelFile.toString(),
                    formattedEndDate,
                    lclid,
                    trainingParametersManager.getLearningRate().toString(),
                    trainingParametersManager.getBatchSize().toString(),
                    trainingParametersManager.getEpochs().toString(),
                    trainingParametersManager.getEarlyStoppingPatience().toString(),
                    trainingParametersManager.getFineTuneLayers().toString(),
                    "first_training"
            );
        } else {
            command = List.of(
                    pathManager.getShortPythonExecutablePath(),
                    scriptFile.getAbsolutePath(),
                    dataJsonPath,
                    modelFile.toString(),
                    formattedEndDate,
                    lclid,
                    Double.toString(0.001),
                    Integer.toString(32),
                    Integer.toString(10),
                    Integer.toString(5),
                    Integer.toString(3),
                    "first_training"
            );
        }

        workHelper(command, currentDir, lclid, formattedEndDate, aggregationType);
    }

    /**
     * Signals readiness to the parent fog node.
     *
     * @param lclid           The logical cluster ID.
     * @param aggregationType The aggregation type used for training.
     */
    private void signalReadinessToFog(String lclid, AggregationType aggregationType) {
        RestTemplate restTemplate = new RestTemplate();
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);

        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        body.add("lclid", lclid);

        HttpEntity<MultiValueMap<String, String>> requestEntity = new HttpEntity<>(body, headers);

        String url = aggregationType.equals(AggregationType.GENETIC) ? deviceManager.getApis().get("ready") :
                deviceManager.getApis().get("ready-favg");

        System.out.println("Sending readiness signal to fog...");
        System.out.println("API: " + url);
        System.out.println("Request Body: " + body);

        try {
            ResponseEntity<String> response = restTemplate.postForEntity(url, requestEntity, String.class);
            System.out.println("Response: " + response.getStatusCode() + " " + response.getBody());
        } catch (HttpClientErrorException e) {
            System.out.println("Error: " + e.getStatusCode() + " " + e.getResponseBodyAsString());
        }
    }

    /**
     * Waits for a signal from the parent fog node.
     *
     * @param lclid           The logical cluster ID.
     * @param aggregationType The aggregation type used for training.
     */
    private void waitForFogSignal(String lclid, AggregationType aggregationType) {
        boolean proceed = false;
        RestTemplate restTemplate = new RestTemplate();

        String url = aggregationType.equals(AggregationType.GENETIC) ? deviceManager.getApis().get("proceed")
                .replace("lclid", lclid) : deviceManager.getApis().get("proceed-favg")
                .replace("lclid", lclid);

        while (!proceed) {
            System.out.println("proceed: " + proceed);
            try {
                ResponseEntity<Boolean> response = restTemplate.getForEntity(url, Boolean.class);
                System.out.println(lclid + ": " + response.getBody());
                proceed = (response.getBody() != null) && response.getBody();
                if (!proceed) {
                    Thread.sleep(1000);
                }
            } catch (Exception e) {
                throw new RuntimeException("Error while waiting for fog signal: " + e.getMessage());
            }
        }
    }

    /**
     * Extracts a valid date from a string.
     *
     * @param dateStr The input string containing the date.
     * @return A string representing the extracted date in the format `yyyy-MM-dd`.
     * @throws IllegalArgumentException If no valid date is found in the input string.
     */
    private String extractDate(String dateStr) {
        Pattern pattern = Pattern.compile("\\d{4}-\\d{2}-\\d{2}");
        Matcher matcher = pattern.matcher(dateStr);
        if (matcher.find()) {
            return matcher.group(0);
        }
        throw new IllegalArgumentException("No valid date found in string: " + dateStr);
    }

    /**
     * Helper method to execute a command and process the results.
     *
     * @param command          The command to execute.
     * @param currentDir       The current directory for execution.
     * @param lclid            The logical cluster ID.
     * @param endDate          The end date for the process.
     * @param aggregationType  The aggregation type used for training.
     * @throws IOException          If an error occurs during execution or file handling.
     * @throws InterruptedException If the thread is interrupted while waiting for the command to finish.
     */
    private void workHelper(List<String> command, Path currentDir, String lclid, String endDate,
                            AggregationType aggregationType)
            throws IOException, InterruptedException {
        String bestModelPath;
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        // Set the PYTHONPATH environment variable to include the script directory
        processBuilder.environment().put(pathManager.getPythonEnvVar(), pathManager.getScriptsDirectoryPath());
        processBuilder.redirectErrorStream(true);

        //Map<String, Double> result = runPythonString(processBuilder);
        Map<String, Map<String, Double>> result = runPythonString(processBuilder);
        Map<String, Double> originalModelPerformance = result.get("original_model");
        double originalLoss = originalModelPerformance.get("loss");

        Map<String, Double> retrainedModelPerformance = result.get("retrained_model");
        double retrainedLoss = retrainedModelPerformance.get("loss");
        double retrainedMae = retrainedModelPerformance.get("mae");
        double retrainedR2 = retrainedModelPerformance.get("r2");
        double retrainedRmse = retrainedModelPerformance.get("rmse");

        dataManager.addPerformanceResult(originalLoss, retrainedLoss, endDate);

        Path csvFilePath = Paths.get(pathManager.getMetricsCSVPath());
        boolean csvExists = Files.exists(csvFilePath);

        if (!csvExists) {
            try (BufferedWriter writer = Files.newBufferedWriter(csvFilePath, StandardOpenOption.CREATE)) {
                writer.write("date, mae, r2, rmse\n");
                writer.write(String.format("%s, %.6f, %.6f, %.6f\n", endDate, retrainedMae, retrainedR2, retrainedRmse));
            }
        } else {
            try (BufferedWriter writer = Files.newBufferedWriter(csvFilePath, StandardOpenOption.APPEND)) {
                writer.write(String.format("%s,%.6f,%.6f,%.6f\n", endDate, retrainedMae, retrainedR2, retrainedRmse));
            }
        }

        if (aggregationType.equals(AggregationType.GENETIC)) {
            Path modelsDirectory = Paths.get(pathManager.getModelsDirectory()).toAbsolutePath();
            Path fogModelPath = modelsDirectory.resolve(Objects.requireNonNull(pathManager.getEnhancedModelPath()));

            if (retrainedLoss > originalLoss) {
                bestModelPath = fogModelPath.toString();
            } else {
                bestModelPath = currentDir.resolve(pathManager.getLocalModelPath(lclid, endDate)).toString();
            }
        } else {
            bestModelPath = currentDir.resolve(pathManager.getLocalModelPath(lclid, endDate)).toString();
        }

        System.out.println("best model path: " + bestModelPath);
        sendResultToFog(originalLoss, retrainedLoss, bestModelPath, lclid, aggregationType);
        dataManager.savePerformanceToJson();

        System.out.println("Edge incoming traffic: " + edgeTrafficManager.getIncomingTrafficOverIterations());
        System.out.println("Edge outgoing traffic: " + edgeTrafficManager.getOutgoingTrafficOverIterations());
    }

    /**
     * Handles the genetics validation process for the edge model by executing a training script.
     * <p>
     * Depending on the number of dates provided, it either trains on a daily or a period dataset.
     * It also processes the fog model file and runs the genetics training command.
     * </p>
     *
     * @param file       The model file received from the fog node.
     * @param date       The date(s) for training (single date or start-end period).
     * @param lclid      The logical cluster ID of the edge.
     * @param learningRate The learning rate for training.
     * @param batchSize  The batch size for training.
     * @param epochs     The number of epochs for training.
     * @param patience   The patience value for early stopping.
     * @param fineTune   The number of layers to fine-tune.
     * @return A {@link ResponseEntity} containing the performance result of the training.
     * @throws IOException          If an error occurs during file or data processing.
     * @throws InterruptedException If the thread is interrupted during execution.
     */
    public ResponseEntity<?> doGeneticsValidation(MultipartFile file, List<String> date, String lclid, String learningRate,
                                                  String batchSize, String epochs, String patience, String fineTune)
            throws IOException, InterruptedException {
        String dataJsonPath;
        if (date.size() == 2) {
            String formattedStartDate = extractDate(date.get(0));
            String formattedEndDate = extractDate(date.get(1));
            dataJsonPath = dataManager.providePeriodData(formattedStartDate, formattedEndDate, lclid);
            edgeTrafficManager.setCurrentWorkingDate(formattedEndDate);
        } else {
            String formattedDate = extractDate(date.get(0));
            edgeTrafficManager.setCurrentWorkingDate(formattedDate);
            dataJsonPath = dataManager.provideDailyData(formattedDate, lclid);
        }

        waitForDailyData(dataJsonPath);

        getFogModelFile(file);

        //File scriptFile = extractScript(pathManager.getGeneticsTrainingScriptPath());
        File scriptFile = new File(pathManager.getGeneticsTrainingScriptPath());

        List<String> command = List.of(
                pathManager.getShortPythonExecutablePath(),
                scriptFile.getAbsolutePath(),
                dataJsonPath,
                learningRate,
                batchSize,
                epochs,
                patience,
                fineTune
        );
        System.out.println("genetics command: " + command);
        Double performance = geneticsWorker(command);

        if (date.size() == 2) {
            String formattedEndDate = extractDate(date.get(1));
            if (!dataManager.existsPerformanceByDate(formattedEndDate) && dataManager.getPerformanceResultList().isEmpty()) {
                dataManager.addPerformanceResult(0.0, performance, formattedEndDate);
                dataManager.savePerformanceToJson();
            }
        } else {
            dataManager.loadPerformanceFromJson();
            String formattedDate = extractDate(date.get(0));
            if (!dataManager.existsPerformanceByDate(formattedDate)) {
                dataManager.addPerformanceResult(0.0, performance, formattedDate);
                dataManager.savePerformanceToJson();
            }
        }

        return ResponseEntity.ok(performance);
    }

    /**
     * Executes a Python script for genetics training and retrieves the performance metrics.
     *
     * @param command The command to execute.
     * @return The performance of the retrained model.
     * @throws IOException          If an error occurs during execution or file handling.
     * @throws InterruptedException If the thread is interrupted while waiting for the command to finish.
     */
    private Double geneticsWorker(List<String> command) throws IOException, InterruptedException {
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.environment().put(pathManager.getPythonEnvVar(), pathManager.getScriptsDirectoryPath());
        processBuilder.redirectErrorStream(true);

        dataManager.loadPerformanceFromJson();

        Map<String, Map<String, Double>> result = runPythonString(processBuilder);
        Map<String, Double> retrainedModelPerformance = result.get("retrained_model");
        dataManager.savePerformanceToJson();

        Double retrainedPerformance = retrainedModelPerformance.get("loss");
        System.out.println("Retrained performance: " + retrainedPerformance);

        return retrainedPerformance;
    }

    /**
     * Waits for daily data to be available before proceeding.
     *
     * @param dataJsonPath The path to the data file.
     * @throws IOException          If the data file is not available within the timeout period.
     * @throws InterruptedException If the thread is interrupted while waiting for the data file.
     */
    private void waitForDailyData(String dataJsonPath) throws IOException, InterruptedException {
        while (dataJsonPath == null || !new File(dataJsonPath).exists()) {
            // Check if timeout has been reached
            System.out.println("waiting for the daily data...");
            if (System.currentTimeMillis() - dataManager.getStartTime() > dataManager.getTimeout()) {
                throw new IOException("Timeout waiting for the dataJsonPath file to be created");
            }
            // Wait for a short period before checking again
            Thread.sleep(500);
        }
    }

    /**
     * Saves the provided fog model file to the model's directory.
     *
     * @param file The fog model file to save.
     * @throws IOException If an error occurs during file handling.
     */
    private void getFogModelFile(MultipartFile file) throws IOException {
        Path modelsDirectory = Paths.get(pathManager.getModelsDirectory()).toAbsolutePath();
        Path fogModelPath = modelsDirectory.resolve(Objects.requireNonNull(pathManager.getEnhancedModelPath()));
        Files.write(fogModelPath, file.getBytes());
    }

    /**
     * Retrieves the list of incoming edge traffic metrics.
     *
     * @return A response containing the list of incoming edge traffic metrics.
     */
    public ResponseEntity<List<Double>> requestIncomingEdgeTraffic() {
        edgeTrafficManager.updateIncomingTrafficJsonFile();
        return ResponseEntity.ok(edgeTrafficManager.getIncomingTrafficOverIterations()
                .stream().map(EdgeTraffic::getTraffic).toList());
    }

    /**
     * Retrieves the list of outgoing edge traffic metrics.
     *
     * @return A response containing the list of outgoing edge traffic metrics.
     */
    public ResponseEntity<List<Double>> requestOutgoingEdgeTraffic() {
        edgeTrafficManager.updateOutgoingTrafficJsonFile();
        return ResponseEntity.ok(edgeTrafficManager.getOutgoingTrafficOverIterations()
                .stream().map(EdgeTraffic::getTraffic).toList());
    }

    /**
     * Retrieves the performance results of the edge model.
     *
     * @return A response containing the performance results.
     */
    public ResponseEntity<?> requestPerformanceResult() {
        dataManager.loadPerformanceFromJson();
        return ResponseEntity.ok(dataManager.getPerformanceResultList());
    }

    /**
     * Sets the working date for the edge node.
     *
     * @param date The working date to set.
     * @return A response confirming the working date was set successfully.
     */
    public ResponseEntity<?> setWorkingDate(String date) {
        System.out.println("current working date: " + edgeTrafficManager.getCurrentWorkingDate());
        edgeTrafficManager.setCurrentWorkingDate(date);
        System.out.println("set current working date: " + edgeTrafficManager.getCurrentWorkingDate());
        return ResponseEntity.ok("Working date saved to " + date);
    }

    /**
     * Evaluates the global model using the provided file and parameters.
     *
     * @param file  The global model file to evaluate.
     * @param date  The evaluation date(s).
     * @param lclid The logical cluster ID.
     * @return A response confirming the evaluation was successful.
     * @throws IOException          If an error occurs during file handling.
     * @throws InterruptedException If the thread is interrupted during evaluation.
     */
    public ResponseEntity<?> evaluateGlobalModel(MultipartFile file, List<String> date, String lclid) throws IOException, InterruptedException {
        String formattedTrainingDate = extractDate(date.get(0));
        System.out.println("Formatted start date: " + formattedTrainingDate + " -> " + lclid);
        String dataJsonPath;
        dataJsonPath = dataManager.provideDailyData(formattedTrainingDate, lclid);

        Path modelsDirectory = Paths.get(pathManager.getModelsDirectory()).toAbsolutePath();
        Path cloudModelPath = modelsDirectory.resolve(Objects.requireNonNull(pathManager.getEnhancedModelPath()));
        Files.write(cloudModelPath, file.getBytes());
        File cloudModel = new File(String.valueOf(cloudModelPath));
        System.out.println("edge evaluate cloud model at path: " + cloudModelPath);

        File scriptFile = new File(pathManager.getModelEvaluateScriptPath());

        List<String> command = List.of(
                pathManager.getShortPythonExecutablePath(),
                scriptFile.getAbsolutePath(),
                dataJsonPath,
                cloudModel.toString(),
                formattedTrainingDate
        );

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.environment().put(pathManager.getPythonEnvVar(), pathManager.getScriptsDirectoryPath());
        processBuilder.redirectErrorStream(true);

        Process process = processBuilder.start();

        StringWriter outputWriter = new StringWriter();
        StringWriter errorWriter = new StringWriter();

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
             BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line + "\n");
                outputWriter.write(line + "\n");
            }
            while ((line = errorReader.readLine()) != null) {
                System.out.println(line + "\n");
                errorWriter.write(line + "\n");
            }
        }

        int exitCode = process.waitFor();
        System.out.println("Python script model_evaluation exited with code: " + exitCode);

        return ResponseEntity.ok("The cloud model was evaluated successfully!");
    }
}