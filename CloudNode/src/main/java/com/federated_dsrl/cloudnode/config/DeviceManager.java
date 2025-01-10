package com.federated_dsrl.cloudnode.config;

import com.federated_dsrl.cloudnode.entity.EdgeTuple;
import lombok.Getter;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Manages mappings between fog nodes and their associated edge devices.
 * This class provides a centralized configuration for fog and edge devices
 * within the distributed system.
 */
@Getter
@Component
public class DeviceManager {

    /**
     * Maps fog node identifiers to their corresponding fog names.
     * <p>Key: String representing the fog identifier (e.g., "230").</p>
     * <p>Value: String representing the fog name (e.g., "fog2").</p>
     */
    private final Map<String, String> fogsMap;

    /**
     * Maps edge node identifiers to their corresponding edge names and MAC addresses.
     * <p>Key: String representing the edge identifier (e.g., "221").</p>
     * <p>Value: List of Strings containing the edge name (e.g., "edge2")
     * and the MAC address (e.g., "MAC000434").</p>
     */
    private final Map<String, List<String>> edgesMap;

    /**
     * Maps fog nodes to their associated edge devices.
     * <p>Key: String representing the fog identifier.</p>
     * <p>Value: List of {@link EdgeTuple} objects representing associated edge devices.</p>
     */
    private final Map<String, List<EdgeTuple>> associatedEdgesToFogMap;

    /**
     * Initializes the `DeviceManager` with default mappings for fog nodes, edge nodes,
     * and their associations.
     */
    public DeviceManager() {
        // Initialize fogsMap with fog identifiers and names
        fogsMap = new HashMap<>();
        fogsMap.put("230", "fog2");
        fogsMap.put("231", "fog3");

        // Initialize edgesMap with edge identifiers, names, and MAC addresses
        edgesMap = new HashMap<>();
        edgesMap.put("221", List.of("edge2", "MAC000434"));
        edgesMap.put("222", List.of("edge3", "MAC004505"));
        edgesMap.put("223", List.of("edge4", "MAC002451"));
        edgesMap.put("224", List.of("edge5", "MAC002163"));
        edgesMap.put("225", List.of("edge6", "MAC001441"));
        edgesMap.put("226", List.of("edge7", "MAC001326"));

        // Initialize associatedEdgesToFogMap as an empty map
        associatedEdgesToFogMap = new HashMap<>();
    }
}
