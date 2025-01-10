package org.federated_dsrl.edgenode.config;

import lombok.Getter;
import org.springframework.stereotype.Component;

/**
 * Manages paths and configurations for file locations, scripts, and environment variables.
 */
@Getter
@Component
public class PathManager {

    // Path constants
    private final String clientEnvScriptPath;
    private final String scriptsDirectoryPath;
    private final String geneticsTrainingScriptPath;
    private final String pythonEnvVar;
    private final String globalModelFileName;
    private final String shortPythonExecutablePath;
    private final String createInitLSTMModelFilePath;
    private final String pythonVirtualExecutablePath;
    private final String selectDataScriptPath;
    private final String dateTimeFormat;
    private final String selectMultipleDataScriptPath;
    private final String utilsScriptPath;
    private final String modelsDirectory;
    private final String enhancedModelPath;
    private final String metricsCSVPath;
    private final String modelEvaluateScriptPath;

    /**
     * Initializes the path manager with default configurations.
     */
    public PathManager() {
        this.clientEnvScriptPath = "/app/scripts/model_usage/client_env.py";
        this.scriptsDirectoryPath = "/app/scripts";
        this.geneticsTrainingScriptPath = "/app/scripts/model_usage/genetics_training.py";
        this.pythonEnvVar = "PYTHONPATH";
        this.globalModelFileName = "global_model.keras";
        this.shortPythonExecutablePath = "python3";
        this.createInitLSTMModelFilePath = "/app/scripts/model_usage/create_init_lstm_model.py";
        this.pythonVirtualExecutablePath = "/opt/venv/bin/python3";
        this.selectDataScriptPath = "/app/scripts/data_usage/select_data.py";
        this.dateTimeFormat = "yyyy-MM-dd HH:mm:ss";
        this.selectMultipleDataScriptPath = "/app/scripts/data_usage/select_multiple_data.py";
        this.utilsScriptPath = "/app/scripts/utils/utils.py";
        this.modelsDirectory = "/app/models/";
        this.enhancedModelPath = "enhanced_model.keras";
        this.metricsCSVPath = "/app/cache_json/metrics.csv";
        this.modelEvaluateScriptPath = "/app/scripts/model_usage/model_evaluation.py";
    }

    /**
     * Constructs the local model file path for a specific logical cluster ID and date.
     *
     * @param lclid the logical cluster ID.
     * @param date  the date in the format "yyyy-MM-dd".
     * @return the constructed path for the local model file.
     */
    public String getLocalModelPath(String lclid, String date) {
        return "models/local_model_" + lclid + "_" + date.replace("-", "_") + ".keras";
    }

    /**
     * Constructs the path to the CSV file associated with a specific logical cluster ID.
     *
     * @param lclid the logical cluster ID.
     * @return the constructed path for the CSV file.
     */
    public String getCSVFilePath(String lclid) {
        return "/app/data/collected/" + lclid + ".csv";
    }

    /**
     * Constructs the path to the data JSON file for a specific logical cluster ID.
     *
     * @param lclid the logical cluster ID.
     * @return the constructed path for the JSON file.
     */
    public String getDataJsonPath(String lclid) {
        return "/app/data/clients/output_data_" + lclid + ".csv";
    }

    /**
     * Constructs the energy data file name for a specific logical cluster ID.
     *
     * @param lclid the logical cluster ID.
     * @return the constructed energy data file name.
     */
    public String getEnergyDataFileName(String lclid) {
        return "energy_data_" + lclid;
    }
}
