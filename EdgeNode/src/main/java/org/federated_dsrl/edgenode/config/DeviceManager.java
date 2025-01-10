package org.federated_dsrl.edgenode.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

/**
 * Manages API endpoint mappings for the edge node system.
 * <p>
 * The {@code DeviceManager} class dynamically constructs and updates API mappings
 * based on the provided host information. It facilitates communication with
 * fog nodes by maintaining a central map of API endpoints.
 * </p>
 */
@Setter
@Getter
@Component
public class DeviceManager {

    /**
     * A map containing API endpoint names as keys and their corresponding URLs as values.
     */
    private final Map<String, String> apis;

    /**
     * Constructs a new {@code DeviceManager} and initializes the API mappings map.
     */
    public DeviceManager() {
        this.apis = new HashMap<>();
    }

    /**
     * Updates the API endpoint mappings based on the provided host address.
     *
     * <p>This method constructs API endpoint URLs using the given host address
     * and stores them in the {@link #apis} map with descriptive keys.</p>
     *
     * <ul>
     *     <li>{@code receive} - URL for receiving edge models from the fog node.</li>
     *     <li>{@code receive-favg} - URL for receiving models during the FAVG aggregation process.</li>
     *     <li>{@code check} - URL to check if the cooling operation is active.</li>
     *     <li>{@code ready} - URL to mark the edge as ready for operation.</li>
     *     <li>{@code ready-favg} - URL to mark the edge as ready during the FAVG aggregation process.</li>
     *     <li>{@code proceed} - URL to check if the edge should proceed with its tasks.</li>
     *     <li>{@code proceed-favg} - URL to check if the edge should proceed during the FAVG aggregation process.</li>
     * </ul>
     *
     * @param host the host address (last byte of the IP address) of the fog node.
     */
    public void updateMapping(String host) {
        apis.put("receive", "http://192.168.2." + host + ":8080/fog/receive-edge-model");
        apis.put("receive-favg", "http://192.168.2." + host + ":8080/fog-favg/receive-edge-model");
        apis.put("check", "http://192.168.2." + host + ":8080/fog/is-cooling-op");
        apis.put("ready", "http://192.168.2." + host + ":8080/fog/edge-ready");
        apis.put("ready-favg", "http://192.168.2." + host + ":8080/fog-favg/edge-ready");
        apis.put("proceed", "http://192.168.2." + host + ":8080/fog/edge-should-proceed/lclid");
        apis.put("proceed-favg", "http://192.168.2." + host + ":8080/fog-favg/edge-should-proceed/lclid");
    }
}
