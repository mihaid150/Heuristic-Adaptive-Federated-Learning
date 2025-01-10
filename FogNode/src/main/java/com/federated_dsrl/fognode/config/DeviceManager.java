package com.federated_dsrl.fognode.config;

import com.federated_dsrl.fognode.entity.EdgeEntity;
import com.federated_dsrl.fognode.utils.FogServiceUtils;
import lombok.Getter;
import lombok.Setter;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Manages and caches the physical nodes (edge devices) and their associated APIs
 * used across the federated network. Provides functionality to add and manage
 * edge devices and their communication endpoints.
 */
@Component
@Getter
public class DeviceManager {
    /**
     * A list of all cached edge devices in the network.
     */
    private final List<EdgeEntity> edges;

    /**
     * The host address of the cloud node in the federated network.
     */
    @Setter
    private String cloudHost;

    /**
     * The name of the current fog node in the federated network.
     */
    @Setter
    private String fogName;

    /**
     * Initializes a new instance of the {@code DeviceManager} class with an empty list of edge devices.
     */
    public DeviceManager() {
        this.edges = new ArrayList<>();
    }

    /**
     * Adds a new edge device to the cache and associates it with its corresponding APIs.
     * If the edge device already exists, it is not added again.
     *
     * @param name            The name of the edge device.
     * @param host            The host address of the edge device.
     * @param lclid           The local client ID (lclid) of the edge device, representing the class from the dataset.
     * @param fogServiceUtils A utility instance used to log progress and provide additional functionality.
     */
    public void addEdge(String name, String host, String lclid, FogServiceUtils fogServiceUtils) {
        HashMap<String, String> apis = new HashMap<>();

        // API for receiving a fog model
        apis.put("receive", "http://192.168.2." + host + ":8080/edge/receive-fog-model");

        // API for informing the edge about the parent fog host address
        apis.put("host", "http://192.168.2." + host + ":8080/edge/parent-fog");

        // API for checking whether the edge is waiting
        apis.put("wait", "http://192.168.2." + host + ":8080/edge/is-waiting");

        // API for enabling genetic search of hyperparameters
        apis.put("genetics", "http://192.168.2." + host + ":8080/edge/genetics-training");

        // API for sending the current iteration hyperparameters to the edge
        apis.put("params", "http://192.168.2." + host + ":8080/edge/receive-parameters");

        // API for informing the edge about the current iteration date
        apis.put("date", "http://192.168.2." + host + ":8080/edge/set-working-date");

        // API for informing the edge about the current simulation model type
        apis.put("model", "http://192.168.2." + host + ":8080/edge/set-model-type");

        // Build the EdgeEntity object with the provided details
        EdgeEntity edgeEntity = EdgeEntity.builder()
                .name(name)
                .lclid(lclid)
                .endpoints(apis)
                .build();

        // Add the edge to the list if it doesn't already exist
        if (!edges.contains(edgeEntity)) {
            fogServiceUtils.logInfo("Adding edge with name: " + name + ", host: " + host + ", lclid: " + lclid);
            this.edges.add(edgeEntity);
        }
    }
}
