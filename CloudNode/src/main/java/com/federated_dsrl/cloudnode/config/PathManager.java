package com.federated_dsrl.cloudnode.config;

import lombok.Getter;
import org.springframework.stereotype.Component;

@Getter
@Component
public class PathManager {
    private final String createInitLSTMModelScript;
    private final String cloudStatisticsScriptPath;
    private final String fogStatisticsScriptPath;
    private final String trafficStatisticsScriptPath;
    private final String performanceStatisticsScriptPath;
    private final String python3ExecutablePath;
    private final String globalModelPath;
    private final String cloudDirectoryPath;
    private final String resultsFilePath;
    private final String aggregatedModelPath;
    private final String aggregatedFavgModelPath;

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

    public String getFogModelFileName(String fogName) {
        return "fog_model_" + fogName + ".keras";
    }
}
