package com.federated_dsrl.fognode.utils;

import com.federated_dsrl.fognode.config.AggregationType;
import com.federated_dsrl.fognode.config.PathManager;
import com.federated_dsrl.fognode.entity.EdgeEntity;
import com.federated_dsrl.fognode.tools.simulated_annealing.FogCoolingSchedule;
import lombok.RequiredArgsConstructor;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpHeaders;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

/**
 * Handles operations related to fog model files, including saving, sending, and aggregating models.
 */
@Component
@RequiredArgsConstructor
public class ModelFileHandler {
    private final PathManager pathManager;
    private final HttpRequestHandler httpRequestHandler;
    private final FogCoolingSchedule fogCoolingSchedule;

    /**
     * Validates the given model file.
     *
     * @param modelFile the model file to validate.
     */
    public void validateModelFile(MultipartFile modelFile) {
        if (modelFile == null || modelFile.isEmpty()) {
            throw new IllegalArgumentException("Model file is null or empty");
        }
    }

    /**
     * Saves the given fog model file.
     *
     * @param fogModel the fog model file to save.
     * @throws IOException if an error occurs during file saving.
     */
    public void saveFogModel(MultipartFile fogModel) throws IOException {
        validateModelFile(fogModel);
        Path modelsDirectory = Paths.get(pathManager.getModelsDirectory()).toAbsolutePath();
        if (!Files.exists(modelsDirectory)) {
            Files.createDirectories(modelsDirectory);
        }
        Path fogModelPath = modelsDirectory.resolve(pathManager.getFogModelPath());
        Files.write(fogModelPath, fogModel.getBytes());
        logInfo("Saved fog model at: " + fogModelPath.toAbsolutePath());
    }


    /**
     * Saves the given edge model file.
     *
     * @param edgeModel the edge model file to save.
     * @param lclid     the LCID (logical cluster ID) associated with the model.
     * @param fogServiceUtils the utility class for logging.
     * @return the path where the edge model file was saved.
     * @throws IOException if an error occurs during file saving.
     */
    public Path saveEdgeModel(MultipartFile edgeModel, String lclid, FogServiceUtils fogServiceUtils) throws IOException {
        validateModelFile(edgeModel);
        Path modelsDirectory = Paths.get(pathManager.getModelsDirectory()).toAbsolutePath();
        if (!Files.exists(modelsDirectory)) {
            Files.createDirectories(modelsDirectory);
            fogServiceUtils.logInfo("Created models directory at: " + modelsDirectory);
        } else {
            fogServiceUtils.logInfo("Models directory already exists at: " + modelsDirectory);
        }
        Path edgeModelPath = modelsDirectory.resolve(pathManager.getBestEdgeModelFileName(lclid));
        Files.write(edgeModelPath, edgeModel.getBytes());
        fogServiceUtils.logInfo("Edge model saved at: " + edgeModelPath);
        return edgeModelPath;

    }

    /**
     * Retrieves the path of the fog model file.
     *
     * @return the path of the fog model file.
     */
    public Path getFogModelPath() {
        return Paths.get(pathManager.getModelsDirectory(), pathManager.getFogModelPath());
    }

    /**
     * Sends a fog model to an edge with retry logic.
     *
     * @param edge             the edge to send the model to.
     * @param fogModelPath     the path of the fog model file.
     * @param date             the associated date.
     * @param isInitialTraining whether this is the initial training step.
     * @param aggregationType  the type of aggregation.
     * @throws IOException if the model could not be sent after retries.
     */
    public void sendCloudModelToEdgesWithRetry(EdgeEntity edge, Path fogModelPath, List<String> date,
                                               Boolean isInitialTraining, AggregationType aggregationType)
            throws IOException {
        int attempt = 0;
        boolean success = false;
        while (attempt < 3 && !success) {
            try {
                sendCloudModelToEdges(edge, fogModelPath, date, isInitialTraining, aggregationType);
                success = true;
            } catch (IOException e) {
                attempt++;
                logError("Failed to send cloud model to " + edge.getName() + " on attempt " + attempt + ". Retrying...");
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new IOException("Retry interrupted", ie);
                }
            }
        }
        if (!success) {
            throw new IOException("Failed to send cloud model to " + edge.getName() + " after 3 attempts.");
        }
    }


    /**
     * Sends a fog model to an edge.
     *
     * @param edge             the edge to send the model to.
     * @param fogModelPath     the path of the fog model file.
     * @param date             the associated date.
     * @param isInitialTraining whether this is the initial training step.
     * @param aggregationType  the type of aggregation.
     * @throws IOException if an error occurs during the file transfer.
     */
    public void sendCloudModelToEdges(EdgeEntity edge, Path fogModelPath, List<String> date,
                                      Boolean isInitialTraining, AggregationType aggregationType) throws IOException {
        logInfo("Sending cloud model from fog to " + edge.getName());
        File fogModel = new File(fogModelPath.toString());

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new FileSystemResource(fogModel));
        body.add("date", date);
        body.add("lclid", edge.getLclid());
        body.add("initial_training", isInitialTraining.toString());
        body.add("aggregation_type", aggregationType);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(org.springframework.http.MediaType.MULTIPART_FORM_DATA);

        httpRequestHandler.sendPostRequest(edge.getEndpoints().get("receive"), body, headers);
    }

    /**
     * Prepares the command to execute the genetic aggregation script.
     *
     * @param newEdgePerformance the new performance value of the edge.
     * @param oldFogPerformance  the previous performance value of the fog.
     * @param edgeModelPath      the path to the edge model file.
     * @return a list of command-line arguments to execute the genetic aggregation script.
     */
    public List<String> prepareGeneticAggregationCommand(Double newEdgePerformance, Double oldFogPerformance, Path edgeModelPath) {
        File fogModel = new File(Paths.get(pathManager.getModelsDirectory(), pathManager.getFogModelPath()).toString());
        File scriptFile = new File(pathManager.getAggregatedGeneticModelsScriptPath());

        return Arrays.asList(
                pathManager.getPython3ExecutablePath(),
                scriptFile.getAbsolutePath(),
                edgeModelPath.toString(),
                String.valueOf(newEdgePerformance),
                fogModel.getAbsolutePath(),
                String.valueOf(oldFogPerformance),
                String.valueOf(fogCoolingSchedule.getTemperature()),
                String.valueOf(fogCoolingSchedule.getINITIAL_TEMPERATURE()));
    }

    /**
     * Prepares the command to execute the average aggregation script.
     *
     * @param isFirstAggregation {@code true} if this is the first aggregation; otherwise {@code false}.
     * @return a list of command-line arguments to execute the average aggregation script.
     */
    public List<String> prepareAverageAggregationCommand(Boolean isFirstAggregation) {
        File fogModel = new File(Paths.get(pathManager.getModelsDirectory(), pathManager.getFogModelPath()).toString());
        File scriptFile = new File(pathManager.getAggregateFAVGModelScriptPath());
        List<String> edgeModels = new ArrayList<>();
        Path modelsDirectory = Paths.get(pathManager.getModelsDirectory()).toAbsolutePath();
        try (Stream<Path> paths = Files.list(modelsDirectory)) {
             paths.filter(path -> path.getFileName().toString().startsWith("edge_best_model_") &&
                            path.getFileName().toString().endsWith(".keras"))
                     .forEach(path -> edgeModels.add(path.toString()));
        } catch (IOException e) {
            logError("Error getting edge models: " + e.getMessage());
            throw new RuntimeException(e);
        }

        return Arrays.asList(
                pathManager.getPython3ExecutablePath(),
                scriptFile.getAbsolutePath(),
                fogModel.getAbsolutePath(),
                isFirstAggregation.toString(),
                edgeModels.toString()
        );
    }

    /**
     * Executes a script using the provided command and logs output.
     *
     * @param command the command to execute the script.
     * @param key     the identifier for the aggregation type (e.g., "Genetic" or "FAVG").
     */
    public void runScript(List<String> command, String key) {
        try {
            logInfo("Running script with command: " + command);
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.redirectErrorStream(true);
            Process process = processBuilder.start();
            String line;
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            while((line = reader.readLine()) != null) {
                output.append(line);
            }
            System.out.println("run script output: " + output);

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                try (BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()))) {
                    while ((line = errorReader.readLine()) != null) {
                        logError("Error output from " + key + " aggregating models in fog: " + line);
                    }
                }
            }
        } catch (IOException | InterruptedException e) {
            logError("Exception while running " + key + " aggregation script: " + e.getMessage());
            throw new RuntimeException(e);
        }
    }

    /**
     * Sends a fog model to an edge for genetic evaluation.
     *
     * @param edge                  the edge to send the model to.
     * @param fogModelPath          the path of the fog model file.
     * @param date                  the associated date.
     * @param learningRate          the learning rate hyperparameter.
     * @param batchSize             the batch size hyperparameter.
     * @param numberEpochs          the number of epochs hyperparameter.
     * @param earlyStoppingPatience the early stopping patience hyperparameter.
     * @param numberFineTuneLayers  the number of fine-tuning layers hyperparameter.
     * @return the fitness value of the fog model after evaluation.
     */
    public Double sendFogModelToEdgesForGenetics(EdgeEntity edge, Path fogModelPath, List<String> date, Double learningRate,
                                                 Integer batchSize, Integer numberEpochs, Integer earlyStoppingPatience,
                                                 Integer numberFineTuneLayers) {
        File fogModel = getFogModel(fogModelPath);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new FileSystemResource(fogModel));
        body.add("date", date);
        body.add("lclid", edge.getLclid());
        body.add("learning_rate", learningRate);
        body.add("batch_size", batchSize);
        body.add("epochs", numberEpochs);
        body.add("patience", earlyStoppingPatience);
        body.add("fine_tune_layers", numberFineTuneLayers);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(org.springframework.http.MediaType.MULTIPART_FORM_DATA);

        return httpRequestHandler.sendPostRequest(edge.getEndpoints().get("genetics"), body, headers, Double.class);
    }

    /**
     * Counts the number of received edge models in the models directory.
     *
     * @return the number of edge models received.
     */
    public int countReceivedEdgeModels() {
        Path modelsDirectory = Paths.get(pathManager.getModelsDirectory()).toAbsolutePath();
        if (!Files.exists(modelsDirectory)) {
            return 0; // No models received yet
        }

        try (Stream<Path> paths = Files.list(modelsDirectory)) {
            return (int) paths
                    .filter(path -> path.getFileName().toString().startsWith("edge_best_model_") &&
                            path.getFileName().toString().endsWith(".keras"))
                    .count();
        } catch (IOException e) {
            logError("Error counting edge models: " + e.getMessage());
            throw new RuntimeException(e);
        }
    }

    /**
     * Retrieves the fog model file.
     *
     * @param fogModelPath the path of the fog model file.
     * @return the fog model file as a {@link File}.
     */
    private File getFogModel(Path fogModelPath) {
        return new File(fogModelPath.toString());
    }

    /**
     * Logs an informational message.
     *
     * @param message the message to log.
     */
    private void logInfo(String message) {
        System.out.println("INFO: " + message);
    }

    /**
     * Logs an error message.
     *
     * @param message the message to log.
     */
    private void logError(String message) {
        System.err.println("ERROR: " + message);
    }
}