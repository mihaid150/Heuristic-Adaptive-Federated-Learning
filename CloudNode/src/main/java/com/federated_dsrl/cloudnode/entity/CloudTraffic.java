package com.federated_dsrl.cloudnode.entity;

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
 * Represents the traffic statistics for cloud operations in terms of incoming and outgoing traffic.
 * <p>
 * This class manages the current iteration's incoming and outgoing traffic and stores historical traffic data
 * across multiple iterations. It also provides functionality to persist and retrieve traffic data to/from JSON files.
 * </p>
 */
@Component
@Data
public class CloudTraffic {

    private Double currentIterationOutgoingTraffic;
    private Double currentIterationIncomingTraffic;
    private final List<Double> incomingTrafficOverIterations = new ArrayList<>();
    private final List<Double> outgoingTrafficOverIterations = new ArrayList<>();
    private final String INCOMING_TRAFFIC_FILE = "cache_json/incoming_traffic.json";
    private final String OUTGOING_TRAFFIC_FILE = "cache_json/outgoing_traffic.json";

    /**
     * Adds outgoing traffic for the current iteration.
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
     * Adds incoming traffic for the current iteration.
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
     * Resets the outgoing traffic for the current iteration to zero.
     */
    public void resetOutgoingTraffic() {
        this.currentIterationOutgoingTraffic = 0.0;
    }

    /**
     * Resets the incoming traffic for the current iteration to zero.
     */
    public void resetIncomingTraffic() {
        this.currentIterationIncomingTraffic = 0.0;
    }

    /**
     * Stores the outgoing traffic of the current iteration into the historical data list.
     */
    public void storeCurrentIterationOutgoingTraffic() {
        this.outgoingTrafficOverIterations.add(currentIterationOutgoingTraffic);
    }

    /**
     * Stores the incoming traffic of the current iteration into the historical data list.
     */
    public void storeCurrentIterationIncomingTraffic() {
        this.incomingTrafficOverIterations.add(currentIterationIncomingTraffic);
    }

    /**
     * Saves the historical incoming and outgoing traffic data to JSON files.
     */
    public void saveTrafficToJsonFile() {
        Gson gson = new Gson();

        try (FileWriter incomingWriter = new FileWriter(INCOMING_TRAFFIC_FILE);
             FileWriter outgoingWriter = new FileWriter(OUTGOING_TRAFFIC_FILE)) {
            gson.toJson(incomingTrafficOverIterations, incomingWriter);
            gson.toJson(outgoingTrafficOverIterations, outgoingWriter);
            System.out.println("Incoming and outgoing traffic saved successfully to JSON file.");
        } catch (IOException e) {
            System.out.println("Error saving incoming and outgoing traffic: " + e.getMessage());
            throw new RuntimeException(e);
        }
    }

    /**
     * Loads the historical incoming and outgoing traffic data from JSON files.
     * <p>
     * If the files are not found, a warning is logged, and the method exits gracefully.
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

            System.out.println("Incoming and outgoing traffic loaded successfully!");
        } catch (IOException e) {
            System.out.println("Could not load cloud traffic data.");
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
