package com.federated_dsrl.fognode.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

import java.util.HashMap;

/**
 * Represents an edge device in the federated learning framework.
 * <p>
 * Each edge device is identified by its name and local client ID (lclid) and
 * includes a set of endpoints that define its APIs for communication and operations.
 * </p>
 */
@Data
@AllArgsConstructor
@Builder
public class EdgeEntity {

    /**
     * The name of the edge device.
     */
    private String name;

    /**
     * The local client ID (lclid) of the edge device, which uniquely identifies it in the framework.
     */
    private String lclid;

    /**
     * A map of endpoints for the edge device. Each key represents the API name,
     * and the value represents the corresponding URL.
     * <p>
     * Examples of API names:
     * <ul>
     *     <li>"receive": API for receiving the fog model</li>
     *     <li>"host": API for setting the parent fog host address</li>
     *     <li>"genetics": API for enabling genetic training</li>
     * </ul>
     * </p>
     */
    private HashMap<String, String> endpoints;
}
