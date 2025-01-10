package org.federated_dsrl.edgenode.config;

import org.springframework.stereotype.Component;

/**
 * Defines constants for the API endpoints used by the edge node system.
 * <p>
 * This class centralizes the definitions of all endpoint mappings related to the edge node
 * to ensure consistency and maintainability. Each constant represents a specific endpoint path.
 * </p>
 */
@Component
public class EdgeEndpoints {

    /**
     * Base mapping for all edge-related endpoints.
     */
    public static final String EDGE_MAPPING = "/edge";

    /**
     * Endpoint for receiving the fog model from the fog node.
     */
    public static final String RECEIVE_FOG_MODEL = "/receive-fog-model";

    /**
     * Endpoint for identifying the parent fog node associated with the edge.
     */
    public static final String IDENTIFY_PARENT_FOG = "/parent-fog";

    /**
     * Endpoint for initiating genetics-based training on the edge node.
     */
    public static final String GENETICS_TRAINING = "/genetics-training";

    /**
     * Endpoint for receiving parameters from the fog node.
     */
    public static final String RECEIVE_PARAMS = "/receive-parameters";

    /**
     * Endpoint for requesting the incoming traffic statistics of the edge node.
     */
    public static final String REQUEST_INCOMING_EDGE_TRAFFIC = "/request-incoming-edge-traffic";

    /**
     * Endpoint for requesting the outgoing traffic statistics of the edge node.
     */
    public static final String REQUEST_OUTGOING_EDGE_TRAFFIC = "/request-outgoing-edge-traffic";

    /**
     * Endpoint for requesting the performance results of the edge node's model.
     */
    public static final String REQUEST_PERFORMANCE_RESULT = "/request-performance-result";

    /**
     * Endpoint for setting the working date on the edge node.
     */
    public static final String SET_WORKING_DATE = "/set-working-date";

    /**
     * Endpoint for configuring the model type on the edge node.
     */
    public static final String SET_MODEL_TYPE = "/set-model-type";

    /**
     * Endpoint for evaluating a model on the edge node.
     */
    public static final String EVALUATE_MODEL = "/evaluate-model";
}
