package com.federated_dsrl.fognode.entity;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import lombok.Data;
import org.springframework.stereotype.Component;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

/**
 * Manages traffic data for a fog node in the federated learning framework.
 * <p>
 * This class tracks incoming and outgoing traffic for the current iteration, stores
 * historical traffic data over multiple iterations, and provides functionality to
 * save and load traffic data to/from JSON files.
 * </p>
 */
@Component
@Data
public class FogTraffic {

    /**
     * The total outgoing traffic for the current iteration.
     */
    private Double currentIterationOutgoingTraffic;

    /**
     * The total incoming traffic for the current iteration.
     */
    private Double currentIterationIncomingTraffic;

    /**
     * Historical record of incoming traffic over multiple iterations.
     */
    private final List<Double> incomingTrafficOverIterations = new ArrayList<>();

    /**
     * Historical record of outgoing traffic over multiple iterations.
     */
    private final List<Double> outgoingTrafficOverIterations = new ArrayList<>();

    /**
     * The file path for storing incoming traffic data in JSON format.
     */
    private final String INCOMING_TRAFFIC_FILE = "cache_json/incoming_traffic.json";

    /**
     * The file path for storing outgoing traffic data in JSON format.
     */
    private final String OUTGOING_TRAFFIC_FILE = "cache_json/outgoing_traffic.json";

    /**
     * Adds the specified amount of traffic to the current iteration's outgoing traffic.
     *
     * @param traffic the amount of outgoing traffic to add
     */
    public void addOutgoingTraffic(Double traffic) {
        if (this.currentIterationOutgoingTraffic == null) {
            this.currentIterationOutgoingTraffic = traffic;
        } else {
            this.currentIterationOutgoingTraffic += traffic;
        }
    }

    /**
     * Adds the specified amount of traffic to the current iteration's incoming traffic.
     *
     * @param traffic the amount of incoming traffic to add
     */
    public void addIncomingTraffic(Double traffic) {
        if (this.currentIterationIncomingTraffic == null) {
            this.currentIterationIncomingTraffic = traffic;
        } else {
            this.currentIterationIncomingTraffic += traffic;
        }
    }

    /**
     * Resets the current iteration's outgoing traffic to 0.
     */
    public void resetOutgoingTraffic() {
        this.currentIterationOutgoingTraffic = 0.0;
    }

    /**
     * Resets the current iteration's incoming traffic to 0.
     */
    public void resetIncomingTraffic() {
        this.currentIterationIncomingTraffic = 0.0;
    }

    /**
     * Stores the current iteration's outgoing traffic in the historical record.
     */
    public void storeCurrentIterationOutgoingTraffic() {
        this.outgoingTrafficOverIterations.add(currentIterationOutgoingTraffic);
    }

    /**
     * Stores the current iteration's incoming traffic in the historical record.
     */
    public void storeCurrentIterationIncomingTraffic() {
        this.incomingTrafficOverIterations.add(currentIterationIncomingTraffic);
    }

    /**
     * Saves the historical incoming and outgoing traffic data to JSON files.
     * <p>
     * The data is stored in the files specified by {@link #INCOMING_TRAFFIC_FILE}
     * and {@link #OUTGOING_TRAFFIC_FILE}.
     * </p>
     */
    public void saveTrafficToJsonFile() {
        Gson gson = new Gson();
        try (FileWriter incomingWriter = new FileWriter(INCOMING_TRAFFIC_FILE);
             FileWriter outgoingWriter = new FileWriter(OUTGOING_TRAFFIC_FILE)) {
            gson.toJson(incomingTrafficOverIterations, incomingWriter);
            gson.toJson(outgoingTrafficOverIterations, outgoingWriter);
        } catch (IOException e) {
            System.out.println("Error saving incoming and outgoing traffic: " + e.getMessage());
            throw new RuntimeException(e);
        }
    }

    /**
     * Loads the historical incoming and outgoing traffic data from JSON files.
     * <p>
     * The data is read from the files specified by {@link #INCOMING_TRAFFIC_FILE}
     * and {@link #OUTGOING_TRAFFIC_FILE}.
     * </p>
     * <p>
     * If the files do not exist or cannot be read, a message is logged, and the data remains unchanged.
     * </p>
     */
    public void loadTrafficFromJsonFile() {
        Gson gson = new Gson();
        try (FileReader incomingReader = new FileReader(INCOMING_TRAFFIC_FILE);
             FileReader outgoingReader = new FileReader(OUTGOING_TRAFFIC_FILE)) {
            Type trafficType = new TypeToken<List<Double>>() {}.getType();
            List<Double> loadedIncomingTraffic = gson.fromJson(incomingReader, trafficType);
            List<Double> loadedOutgoingTraffic = gson.fromJson(outgoingReader, trafficType);

            incomingTrafficOverIterations.clear();
            outgoingTrafficOverIterations.clear();
            incomingTrafficOverIterations.addAll(loadedIncomingTraffic);
            outgoingTrafficOverIterations.addAll(loadedOutgoingTraffic);
        } catch (IOException e) {
            System.out.println("Could not load fog traffic: " + e.getMessage());
        }
    }

    /**
     * Clears all historical traffic data for both incoming and outgoing traffic.
     */
    public void clearTraffic() {
        incomingTrafficOverIterations.clear();
        outgoingTrafficOverIterations.clear();
    }
}
