package com.federated_dsrl.fognode.tools.simulated_annealing;

import com.federated_dsrl.fognode.config.DeviceManager;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

/**
 * The {@code CloudCoolingSchedule} class is responsible for retrieving the current cloud cooling
 * temperature from a remote cloud server. This temperature is used in simulated annealing
 * calculations for optimization algorithms.
 * <p>
 * The temperature is fetched via an HTTP request to the cloud server's REST API.
 * </p>
 *
 * <p>Example usage:</p>
 * <pre>
 *     CloudCoolingSchedule schedule = new CloudCoolingSchedule(deviceManager);
 *     Double temperature = schedule.getCloudCoolingTemperature();
 * </pre>
 */
@Component
@RequiredArgsConstructor
public class CloudCoolingSchedule {

    private final DeviceManager deviceManager;

    /**
     * Retrieves the current cloud cooling temperature from the cloud server.
     * <p>
     * The method sends an HTTP GET request to the configured cloud server using
     * the host address provided by {@link DeviceManager}. If the server responds
     * successfully, the temperature value is returned. If the response is not
     * successful, a default value of {@code 0.0} is returned.
     * </p>
     *
     * @return The cloud cooling temperature as a {@link Double}. If the request fails,
     * returns {@code 0.0}.
     */
    public Double getCloudCoolingTemperature() {
        System.out.println("Requesting cloud cooling temperature...");
        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<Double> response = restTemplate.getForEntity(
                "http://192.168.2." + deviceManager.getCloudHost() + ":8080/cloud/get-cooling-temperature",
                Double.class
        );

        if (response.getStatusCode().is2xxSuccessful()) {
            System.out.println("Received temperature: " + response.getBody());
            return response.getBody();
        } else {
            System.err.println("Failed to retrieve temperature, returning default 0.0");
            return 0.0;
        }
    }
}
