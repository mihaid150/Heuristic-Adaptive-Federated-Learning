package com.federated_dsrl.fognode.tools.genetic.engine;

import com.federated_dsrl.fognode.config.DeviceManager;
import com.federated_dsrl.fognode.entity.EdgeEntity;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * The {@code EdgeSelector} is responsible for selecting a random edge entity for genetic evaluation
 * of hyperparameters and designating the remaining edges for training using those parameters.
 * <p>
 * This component operates on the edges managed by the {@link DeviceManager} and splits the edges
 * into a single evaluation edge and a list of training edges.
 * </p>
 */
@Component
@RequiredArgsConstructor
@Data
public class EdgeSelector {

    private final DeviceManager deviceManager;
    private List<EdgeEntity> trainingEdges = new ArrayList<>();
    private EdgeEntity evaluateEdge;

    /**
     * Selects a random edge from the list of edges managed by the {@link DeviceManager} as the evaluation edge.
     * <p>
     * The evaluation edge is used for genetic evaluation of hyperparameters, while the remaining edges
     * are designated as training edges for training models with the evaluated parameters.
     * </p>
     * <p>
     * If no edges are available, this method throws an {@link IllegalStateException}.
     * </p>
     *
     * @throws IllegalStateException if no edges are available in the {@link DeviceManager}.
     */
    public void selectRandomEdgeAndGetRemaining() {
        List<EdgeEntity> edges = deviceManager.getEdges();
        trainingEdges.clear();

        if (edges.isEmpty()) {
            throw new IllegalStateException("No edges available in the device manager.");
        }

        Random random = new Random();

        // Select a random index
        int randomIndex = random.nextInt(edges.size());

        // Get the random edge
        evaluateEdge = edges.get(randomIndex);

        // Populate training edges, excluding the evaluation edge
        for (int i = 0; i < edges.size(); i++) {
            if (i != randomIndex) {
                trainingEdges.add(edges.get(i));
            }
        }
    }
}
