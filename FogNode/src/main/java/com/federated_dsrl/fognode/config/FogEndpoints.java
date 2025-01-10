package com.federated_dsrl.fognode.config;

import org.springframework.stereotype.Component;

/**
 * Defines constants for the API endpoints used by the fog node system.
 * <p>
 * This class centralizes the definitions of all endpoint mappings related to the fog node
 * to ensure consistency and maintainability. Each constant represents a specific endpoint path.
 * </p>
 */
@Component
public class FogEndpoints {

    /**
     * Base mapping for all fog-related endpoints.
     */
    public static final String FOG_MAPPING = "/fog";

    /**
     * Base mapping for all fog-related endpoints dealing with FAVG (Federated Averaging).
     */
    public static final String FOG_FAVG_MAPPING = "/fog-favg";

    /**
     * Endpoint for adding an edge node to the fog node system.
     */
    public static final String ADD_EDGE = "/add-edge";

    /**
     * Endpoint for receiving the global model from the cloud.
     */
    public static final String RECEIVE_CLOUD_MODEL = "/receive-global-model";

    /**
     * Endpoint for receiving a model from an edge node.
     */
    public static final String RECEIVE_EDGE_MODEL = "/receive-edge-model";

    /**
     * Endpoint for requesting the fog node's model.
     */
    public static final String REQUEST_FOG_MODEL = "/request-fog-model";

    /**
     * Endpoint for acknowledging the receipt of a model or data from the cloud.
     */
    public static final String ACK_CLOUD = "/ack-cloud";

    /**
     * Endpoint for receiving readiness signals from edge nodes.
     */
    public static final String RECEIVE_READINESS_SIGNAL = "/edge-ready";

    /**
     * Endpoint to check if an edge node should proceed with a given logical cluster ID (LCID).
     *
     */
    public static final String EDGE_SHOULD_PROCEED = "/edge-should-proceed/{lclid}";

    /**
     * Endpoint for requesting the elapsed time list for fog operations.
     */
    public static final String REQUEST_ELAPSED_TIME_LIST = "/request-elapsed-time-list";

    /**
     * Endpoint for requesting the incoming traffic statistics of the fog node.
     */
    public static final String REQUEST_INCOMING_FOG_TRAFFIC = "/request-incoming-fog-traffic";

    /**
     * Endpoint for requesting the outgoing traffic statistics of the fog node.
     */
    public static final String REQUEST_OUTGOING_FOG_TRAFFIC = "/request-outgoing-fog-traffic";

    /**
     * Endpoint for loading the system state on the fog node.
     */
    public static final String LOAD_SYSTEM_STATE = "/load-system-state";

    /**
     * Endpoint for setting or retrieving the type of edge model.
     */
    public static final String EDGE_MODEL_TYPE = "/edge-model-type";
}
