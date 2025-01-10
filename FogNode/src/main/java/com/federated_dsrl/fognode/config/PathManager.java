package com.federated_dsrl.fognode.config;

import lombok.Getter;
import org.springframework.stereotype.Component;

/**
 * Manages file paths and directories used by the fog node system.
 * <p>
 * This class provides centralized management of paths related to models, scripts, and configurations
 * within the application. It ensures consistency and simplifies updates to paths across the system.
 * </p>
 */
@Component
@Getter
public class PathManager {

    /**
     * Directory where models are stored.
     */
    private final String modelsDirectory;

    /**
     * Path to the main fog model file.
     */
    private final String fogModelPath;

    /**
     * Path to the Python 3 executable used for script execution.
     */
    private final String python3ExecutablePath;

    /**
     * Path to the script used for aggregating genetic models.
     */
    private final String aggregatedGeneticModelsScriptPath;

    /**
     * Path to the script used for aggregating FAVG (Federated Averaging) models.
     */
    private final String aggregateFAVGModelScriptPath;

    /**
     * Initializes the {@code PathManager} with default paths for models, scripts, and the Python executable.
     */
    public PathManager() {
        this.modelsDirectory = "/app/models";
        this.fogModelPath = "fog_model.keras";
        this.python3ExecutablePath = "/opt/venv/bin/python3";
        this.aggregatedGeneticModelsScriptPath = "/app/scripts/model_usage/aggregate_genetic_models.py";
        this.aggregateFAVGModelScriptPath = "/app/scripts/model_usage/aggregate_favg_models.py";
    }

    /**
     * Constructs the filename for the best model received from an edge node based on its logical cluster ID (LCID).
     *
     * @param lclid the logical cluster ID of the edge node.
     * @return the filename of the best edge model, formatted as {@code edge_best_model_<lclid>.keras}.
     */
    public String getBestEdgeModelFileName(String lclid) {
        return "edge_best_model_" + lclid + ".keras";
    }
}
