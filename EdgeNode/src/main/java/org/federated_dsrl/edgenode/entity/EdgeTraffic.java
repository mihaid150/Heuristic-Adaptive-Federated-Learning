package org.federated_dsrl.edgenode.entity;

import lombok.Builder;
import lombok.Data;

/**
 * Represents traffic data for an edge node.
 * <p>
 * This entity is used to track and store information about the traffic associated with an edge node
 * for a specific date.
 * </p>
 */
@Data
@Builder
public class EdgeTraffic {
    /**
     * The date associated with the traffic data.
     */
    private String date;

    /**
     * The amount of traffic recorded on the given date.
     */
    private Double traffic;
}

