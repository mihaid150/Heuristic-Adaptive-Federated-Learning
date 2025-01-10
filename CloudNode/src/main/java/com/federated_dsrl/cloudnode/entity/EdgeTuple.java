package com.federated_dsrl.cloudnode.entity;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * Represents a tuple containing information about an edge device in a federated learning network.
 * <p>
 * This class stores essential information about an edge device, such as its name, host, and MAC address.
 * </p>
 */
@Data
@AllArgsConstructor
public class EdgeTuple {

    /**
     * The name of the edge device.
     */
    private String name;

    /**
     * The host address or identifier of the edge device.
     */
    private String host;

    /**
     * The MAC (Media Access Control) address of the edge device.
     */
    private String mac;
}
