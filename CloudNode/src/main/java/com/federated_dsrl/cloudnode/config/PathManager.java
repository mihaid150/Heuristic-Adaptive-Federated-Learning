package com.federated_dsrl.cloudnode.config;

import lombok.Getter;
import org.springframework.stereotype.Component;

/**
 * Manages various file paths and script locations used within the cloud node application.
 * <p>
 * This class centralizes the configuration of paths and ensures consistent access
 * to critical resources such as scripts, model files, and directories.
 * </p>
 */
@Getter
@Component
public class PathManager {

    /** Path to the script for creating the initial LSTM model. */
    private final String createInitLSTMModelScript;

    /** Path to the cloud statistics Python script. */
    private final String cloudStatisticsScriptPath;

    /** Path to the fog statistics Python script. */
    private final String fogStatisticsScriptPath;

    /** Path to the traffic statistics Python script. */
    private final String trafficStatisticsScriptPath;

    /** Path to the performance statistics Python script. */
    private final String performanceStatisticsScriptPath;

    /** Path to the Python 3 executable used by the application. */
    private final String python3ExecutablePath;

    /** Path to the global model file. */
    private final String globalModelPath;

    /** Path to the directory where cloud-related models are stored. */
    private final String cloudDirectoryPath;

    /** Path to the results file used by the cloud node. */
    private final String resultsFilePath;

    /** Path to the Python script for aggregating genetic models. */
    private final String aggregatedModelPath;

    /** Path to the Python script for aggregating FAVG models. */
    private final String aggregatedFavgModelPath;

    /**
     * Default constructor initializing file paths and script locations.
     */
    public PathManager() {
        this.createInitLSTMModelScript = "/app/scripts/model_usage/create_init_lstm_model.py";
        this.python3ExecutablePath = "/opt/venv/bin/python3";
        this.globalModelPath = "/app/models/cloud/global_model.keras";
        this.cloudDirectoryPath = "/app/models/cloud";
        this.resultsFilePath = "/app/models/cloud/results.txt";
        this.aggregatedModelPath = "/app/scripts/model_usage/aggregate_genetic_models.py";
        this.cloudStatisticsScriptPath = "/app/scripts/statistics_usage/cloud_statistics.py";
        this.fogStatisticsScriptPath = "/app/scripts/statistics_usage/fog_statistics.py";
        this.trafficStatisticsScriptPath = "/app/scripts/statistics_usage/traffic_statistics.py";
        this.performanceStatisticsScriptPath = "/app/scripts/statistics_usage/performance_statistics.py";
        this.aggregatedFavgModelPath = "/app/scripts/model_usage/aggregate_favg_models.py";
    }

    /**
     * Generates the filename for a fog model based on the fog's name.
     *
     * @param fogName the name of the fog.
     * @return the generated filename for the fog model.
     */
    public String getFogModelFileName(String fogName) {
        return "fog_model_" + fogName + ".keras";
    }
}
