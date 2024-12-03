package com.federated_dsrl.cloudnode.config;

import org.springframework.stereotype.Component;

@Component
public class CloudEndpoints {
    public static final String CLOUD_MAPPING = "/cloud";
    public static final String INIT_CACHE = "/init/{isCacheActive}/{geneticEvaluationStrategy}/{modelType}";
    public static final String INIT = "/init";
    public static final String CLOUD_FAVG_MAPPING = "/cloud-favg";
    public static final String RECEIVE_FOG_MODEL = "/receive-fog-model";
    public static final String DAILY_FEDERATION_CACHE = "/daily-federation/{isCacheActive}/{date}/{geneticEvaluationStrategy}";
    public static final String DAILY_FEDERATION = "/daily-federation/{date}";
    public static final String CREATE_ELAPSED_TIME_CHART = "/elapsed-time-chart";
    public static final String GET_COOLING_TEMPERATURE = "/get-cooling-temperature";
    public static final String CREATE_ELAPSED_TIME_CHART_FOG_LAYER = "/create-elapsed-time-chart-fog-layer";
    public static final String CREATE_TRAFFIC_CHART = "/create-traffic-chart";
    public static final String CREATE_PERFORMANCE_CHART = "/create-performance-chart";
    public static final String LOAD_SYSTEM_STATE = "/load-system-state";
}
