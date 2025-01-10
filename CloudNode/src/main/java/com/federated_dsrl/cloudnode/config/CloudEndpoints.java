package com.federated_dsrl.cloudnode.config;

import org.springframework.stereotype.Component;

/**
 * Defines API endpoint mappings for the cloud node.
 * This class provides centralized management of REST API endpoint paths
 * used across the cloud-related services.
 */
@Component
public class CloudEndpoints {

    /**
     * Base mapping for all cloud-related endpoints.
     */
    public static final String CLOUD_MAPPING = "/cloud";

    /**
     * Endpoint for initializing the global model with cache settings, genetic evaluation strategy, and model type.
     * <ul>
     *     <li>`isCacheActive`: Determines if cache should be active (true/false).</li>
     *     <li>`geneticEvaluationStrategy`: Specifies the genetic evaluation strategy.</li>
     *     <li>`modelType`: Specifies the type of model to initialize.</li>
     * </ul>
     */
    public static final String INIT_CACHE = "/init/{isCacheActive}/{geneticEvaluationStrategy}/{modelType}";

    /**
     * Endpoint for initializing the global model for federated average (FAVG) aggregation.
     */
    public static final String INIT = "/init";

    /**
     * Base mapping for cloud-related FAVG endpoints.
     */
    public static final String CLOUD_FAVG_MAPPING = "/cloud-favg";

    /**
     * Endpoint for receiving models and results from fog nodes.
     * <ul>
     *     <li>Accepts a multipart file containing the model.</li>
     *     <li>Requires `fog_name` and `current_date` as parameters.</li>
     * </ul>
     */
    public static final String RECEIVE_FOG_MODEL = "/receive-fog-model";

    /**
     * Endpoint for executing a daily federation with cache settings.
     * <ul>
     *     <li>`isCacheActive`: Determines if cache should be active (true/false).</li>
     *     <li>`date`: Specifies the date for federation.</li>
     *     <li>`geneticEvaluationStrategy`: Specifies the genetic evaluation strategy.</li>
     * </ul>
     */
    public static final String DAILY_FEDERATION_CACHE = "/daily-federation/{isCacheActive}/{date}/{geneticEvaluationStrategy}";

    /**
     * Endpoint for executing a daily federation for FAVG.
     * <ul>
     *     <li>`date`: Specifies the date for federation.</li>
     * </ul>
     */
    public static final String DAILY_FEDERATION = "/daily-federation/{date}";

    /**
     * Endpoint for creating a chart of elapsed time for the global cloud process.
     */
    public static final String CREATE_ELAPSED_TIME_CHART = "/elapsed-time-chart";

    /**
     * Endpoint for retrieving the current cooling temperature of the cloud infrastructure.
     */
    public static final String GET_COOLING_TEMPERATURE = "/get-cooling-temperature";

    /**
     * Endpoint for creating a chart of elapsed time for each fog layer process.
     */
    public static final String CREATE_ELAPSED_TIME_CHART_FOG_LAYER = "/create-elapsed-time-chart-fog-layer";

    /**
     * Endpoint for creating a traffic chart that visualizes network traffic across cloud, fog, and edge layers.
     */
    public static final String CREATE_TRAFFIC_CHART = "/create-traffic-chart";

    /**
     * Endpoint for creating a performance chart that evaluates cloud, fog, and edge layers.
     */
    public static final String CREATE_PERFORMANCE_CHART = "/create-performance-chart";

    /**
     * Endpoint for reloading the system state across all nodes.
     */
    public static final String LOAD_SYSTEM_STATE = "/load-system-state";
}

