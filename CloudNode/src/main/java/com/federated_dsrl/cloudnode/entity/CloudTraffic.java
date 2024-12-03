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

@Component
@Data
public class CloudTraffic {
    private Double currentIterationOutgoingTraffic;
    private Double currentIterationIncomingTraffic;
    private final List<Double> incomingTrafficOverIterations = new ArrayList<>();
    private final List<Double> outgoingTrafficOverIterations = new ArrayList<>();
    private final String INCOMING_TRAFFIC_FILE = "cache_json/incoming_traffic.json";
    private final String OUTGOING_TRAFFIC_FILE = "cache_json/outgoing_traffic.json";

    public void addOutgoingTraffic(Double traffic) {
        if (this.currentIterationOutgoingTraffic == null) {
            this.currentIterationOutgoingTraffic = traffic;
        } else {
            this.currentIterationOutgoingTraffic += traffic;
        }
    }

    public void addIncomingTraffic(Double traffic) {
        if (this.currentIterationIncomingTraffic == null) {
            this.currentIterationIncomingTraffic = traffic;
        } else {
            this.currentIterationIncomingTraffic += traffic;
        }
    }

    public void resetOutgoingTraffic() {
        this.currentIterationOutgoingTraffic = 0.0;
    }

    public void resetIncomingTraffic() {
        this.currentIterationIncomingTraffic = 0.0;
    }

    public void storeCurrentIterationOutgoingTraffic() {
        this.outgoingTrafficOverIterations.add(currentIterationOutgoingTraffic);
    }

    public void storeCurrentIterationIncomingTraffic() {
        this.incomingTrafficOverIterations.add(currentIterationIncomingTraffic);
    }

    public void saveTrafficToJsonFile() {
        Gson gson = new Gson();

        try (FileWriter incomingWriter = new FileWriter(INCOMING_TRAFFIC_FILE);
             FileWriter outgoingWriter = new FileWriter(OUTGOING_TRAFFIC_FILE)) {
            gson.toJson(incomingTrafficOverIterations, incomingWriter);
            gson.toJson(outgoingTrafficOverIterations, outgoingWriter);
            System.out.println("Incoming and outgoing traffic saved successfully to json file.");
        } catch (IOException e) {
            System.out.println("Error saving incoming and outgoing traffic: " + e.getMessage());
            throw new RuntimeException(e);
        }
    }

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
            System.out.println("Could not load yet cloud traffic.");
        }
    }

    public void clearTraffic() {
        incomingTrafficOverIterations.clear();
        outgoingTrafficOverIterations.clear();
    }
}
