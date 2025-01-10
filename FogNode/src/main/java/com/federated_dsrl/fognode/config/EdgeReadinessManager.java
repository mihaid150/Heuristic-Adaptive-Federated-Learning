package com.federated_dsrl.fognode.config;

import com.federated_dsrl.fognode.entity.EdgeEntity;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manages the readiness and proceed states of edges in the federated learning framework.
 * <p>
 * This component tracks whether edges have received the model and are ready to retrain,
 * as well as whether they are allowed to proceed with the training process.
 * </p>
 */
@Component
public class EdgeReadinessManager {
    /**
     * A map to track the readiness state of edges. Each edge is identified by its local client ID (lclid),
     * and the readiness state indicates whether the edge is ready to retrain.
     */
    private final Map<String, Boolean> edgeReadinessMap = new ConcurrentHashMap<>();

    /**
     * A map to track the proceed state of edges. Each edge is identified by its local client ID (lclid),
     * and the proceed state indicates whether the edge is allowed to start retraining.
     */
    private final Map<String, Boolean> edgeProceedMap = new ConcurrentHashMap<>();

    /**
     * Marks an edge as ready for retraining.
     *
     * @param lclid the local client ID (lclid) of the edge to be marked as ready
     */
    public void markEdgeAsReady(String lclid) {
        edgeReadinessMap.put(lclid, true);
    }

    /**
     * Initializes the readiness and proceed state maps for the given list of edges.
     * All edges are initially marked as not ready and not allowed to proceed.
     *
     * @param edges the list of edges to initialize
     */
    public void initializeReadiness(List<EdgeEntity> edges) {
        for (EdgeEntity edge : edges) {
            edgeReadinessMap.put(edge.getLclid(), false);
            edgeProceedMap.put(edge.getLclid(), false);
        }
    }

    /**
     * Resets the readiness and proceed state maps for the given list of edges.
     * All edges are marked as not ready and not allowed to proceed.
     *
     * @param edges the list of edges to reset
     */
    public void resetReadiness(List<EdgeEntity> edges) {
        for (EdgeEntity edge : edges) {
            edgeReadinessMap.put(edge.getLclid(), false);
            edgeProceedMap.put(edge.getLclid(), false);
        }
    }

    /**
     * Signals all edges to proceed with the training process.
     *
     * @param edges the list of edges to signal
     */
    public void signalEdgesToProceed(List<EdgeEntity> edges) {
        edges.forEach(edge -> edgeProceedMap.put(edge.getLclid(), true));
    }

    /**
     * Determines whether the specified edge is allowed to proceed with training.
     *
     * @param lclid the local client ID (lclid) of the edge
     * @return {@code true} if the edge is allowed to proceed, {@code false} otherwise
     */
    public boolean shouldProceed(String lclid) {
        return edgeProceedMap.getOrDefault(lclid, false);
    }

    /**
     * Checks whether the specified edge is ready for retraining.
     *
     * @param lclid the local client ID (lclid) of the edge
     * @return {@code true} if the edge is ready, {@code false} otherwise
     */
    public boolean isEdgeReady(String lclid) {
        return edgeReadinessMap.getOrDefault(lclid, false);
    }
}
