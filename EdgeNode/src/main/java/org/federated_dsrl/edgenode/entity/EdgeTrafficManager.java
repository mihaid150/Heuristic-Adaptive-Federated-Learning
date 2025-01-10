package org.federated_dsrl.edgenode.entity;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import lombok.Data;
import org.springframework.stereotype.Component;

import java.io.*;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Manages traffic data (incoming and outgoing) for an edge node, including saving, updating, and
 * exporting traffic data in JSON and CSV formats.
 * <p>
 * The manager handles traffic data over multiple iterations and provides utilities for creating,
 * loading, and updating traffic files in the edge node system.
 * </p>
 */
@Component
@Data
public class EdgeTrafficManager {
    /**
     * The current iteration's outgoing traffic value.
     */
    private Double currentIterationOutgoingTraffic;

    /**
     * The current iteration's incoming traffic value.
     */
    private Double currentIterationIncomingTraffic;

    /**
     * A list of incoming traffic records over multiple iterations.
     */
    private final List<EdgeTraffic> incomingTrafficOverIterations = new ArrayList<>();

    /**
     * A list of outgoing traffic records over multiple iterations.
     */
    private final List<EdgeTraffic> outgoingTrafficOverIterations = new ArrayList<>();

    /**
     * The path to the JSON file storing incoming traffic data.
     */
    private final String INCOMING_TRAFFIC_FILE = "/app/cache_json/incoming_traffic.json";

    /**
     * The path to the JSON file storing outgoing traffic data.
     */
    private final String OUTGOING_TRAFFIC_FILE = "/app/cache_json/outgoing_traffic.json";

    /**
     * The current working date for traffic data.
     */
    private String currentWorkingDate = "2013-07-09";

    /**
     * Gson instance used for serializing and deserializing traffic data.
     */
    private final Gson gson = getGson();

    /**
     * Checks if a given date is already contained in the traffic list.
     *
     * @param edgeTraffic The list of edge traffic records.
     * @param date        The date to check.
     * @return {@code true} if the date is present in the traffic list; otherwise, {@code false}.
     */
    private Boolean isDateContainedInTraffic(List<EdgeTraffic> edgeTraffic, String date) {
        return edgeTraffic.stream().map(EdgeTraffic::getDate).toList().contains(date);
    }

    /**
     * Adds an outgoing traffic record for the current iteration and saves it to a JSON file.
     *
     * @param traffic The outgoing traffic value to add.
     */
    public void addOutgoingTrafficRemastered(Double traffic) {
        if (!Files.exists(Path.of(OUTGOING_TRAFFIC_FILE))) {
            createOutgoingTrafficFile();
        }
        loadOutgoingTrafficFromJsonFile();
        outgoingTrafficOverIterations.add(new EdgeTraffic(currentWorkingDate, traffic));
        saveOutgoingTrafficToJsonFile();
    }

    /**
     * Adds an incoming traffic record for the current iteration and saves it to a JSON file.
     *
     * @param traffic The incoming traffic value to add.
     */
    public void addIncomingTrafficRemastered(Double traffic) {
        if (!Files.exists(Path.of(INCOMING_TRAFFIC_FILE))) {
            createIncomingTrafficFile();
        }
        loadIncomingTrafficFromJsonFile();
        incomingTrafficOverIterations.add(new EdgeTraffic(currentWorkingDate, traffic));
        saveIncomingTrafficToJsonFile();
    }

    /**
     * Updates the incoming traffic JSON file by merging duplicate entries and saving the results.
     */
    public void updateIncomingTrafficJsonFile() {
        loadIncomingTrafficFromJsonFile();
        updateHelper(incomingTrafficOverIterations);
        saveIncomingTrafficToJsonFile();
        loadIncomingTrafficFromJsonFile();
    }

    /**
     * Updates the outgoing traffic JSON file by merging duplicate entries and saving the results.
     */
    public void updateOutgoingTrafficJsonFile() {
        loadOutgoingTrafficFromJsonFile();
        updateHelper(outgoingTrafficOverIterations);
        saveOutgoingTrafficToJsonFile();
        loadOutgoingTrafficFromJsonFile();
    }

    /**
     * Retrieves an edge traffic record by its associated date.
     *
     * @param edgeTrafficList The list of edge traffic records.
     * @param date            The date to search for.
     * @return The {@link EdgeTraffic} record matching the specified date.
     */
    private EdgeTraffic getEdgeTrafficByDate(List<EdgeTraffic> edgeTrafficList, String date) {
        return edgeTrafficList.stream().filter(edgeTraffic -> edgeTraffic.getDate().equals(date))
                .toList().get(0);
    }

    /**
     * Saves incoming traffic data to a JSON file.
     */
    public void saveIncomingTrafficToJsonFile() {
        if (!Files.exists(Path.of(INCOMING_TRAFFIC_FILE))) {
            createIncomingTrafficFile();
        }
        try (FileWriter incomingWriter = new FileWriter(INCOMING_TRAFFIC_FILE)) {
            gson.toJson(incomingTrafficOverIterations, incomingWriter);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Saves outgoing traffic data to a JSON file.
     */
    public void saveOutgoingTrafficToJsonFile() {
        try (FileWriter outgoingWriter = new FileWriter(OUTGOING_TRAFFIC_FILE)) {
            gson.toJson(outgoingTrafficOverIterations, outgoingWriter);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Loads incoming traffic data from a JSON file into memory.
     * If the file does not exist, it creates an empty file.
     */
    public void loadIncomingTrafficFromJsonFile() {
        boolean fileExists = true;
        try (FileReader incomingReader = new FileReader(INCOMING_TRAFFIC_FILE)) {
            Type trafficType = new TypeToken<List<EdgeTraffic>>() {
            }.getType();
            List<EdgeTraffic> loadedIncomingTraffic = gson.fromJson(incomingReader, trafficType);
            incomingTrafficOverIterations.clear();
            incomingTrafficOverIterations.addAll(loadedIncomingTraffic);
            saveIncomingTrafficToCsv(incomingTrafficOverIterations);
        } catch (FileNotFoundException e) {
            fileExists = false;
            createIncomingTrafficFile();
        } catch (IOException e) {
            throw new RuntimeException("[" + new Date() + "] Error loading incoming traffic: " + e.getMessage(), e);
        }
        if (!fileExists) {
            loadIncomingTrafficFromJsonFile();
        }
    }

    /**
     * Loads outgoing traffic data from a JSON file into memory.
     * If the file does not exist, it creates an empty file.
     */
    public void loadOutgoingTrafficFromJsonFile() {
        boolean fileExists = true;
        try (FileReader outgoingReader = new FileReader(OUTGOING_TRAFFIC_FILE)) {
            Type trafficType = new TypeToken<List<EdgeTraffic>>() {
            }.getType();
            List<EdgeTraffic> loadedOutgoingTraffic = gson.fromJson(outgoingReader, trafficType);
            outgoingTrafficOverIterations.clear();
            outgoingTrafficOverIterations.addAll(loadedOutgoingTraffic);
            saveOutgoingTrafficToCsv(outgoingTrafficOverIterations);
        } catch (FileNotFoundException e) {
            fileExists = false;
            createOutgoingTrafficFile();
        } catch (IOException e) {
            throw new RuntimeException("[" + new Date() + "] Error loading outgoing traffic: " + e.getMessage(), e);
        }
        if (!fileExists) {
            loadOutgoingTrafficFromJsonFile();
        }
    }

    /**
     * Saves the provided outgoing traffic data to a CSV file.
     *
     * @param outgoingTraffic The list of outgoing traffic records to save.
     */
    private void saveOutgoingTrafficToCsv(List<EdgeTraffic> outgoingTraffic) {
        String csvFile = "/app/cache_json/outgoing_traffic.csv";  // Path to save the CSV file

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(csvFile))) {
            saveTrafficHelper(outgoingTraffic, writer);
            System.out.println("Outgoing traffic saved to CSV successfully.");
        } catch (IOException e) {
            throw new RuntimeException("[" + new Date() + "] Error saving outgoing traffic to CSV: " + e.getMessage(), e);
        }
    }

    /**
     * Saves the provided incoming traffic data to a CSV file.
     *
     * @param incomingTraffic The list of incoming traffic records to save.
     */
    private void saveIncomingTrafficToCsv(List<EdgeTraffic> incomingTraffic) {
        String csvFile = "/app/cache_json/incoming_traffic.csv";  // Path to save the CSV file

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(csvFile))) {
            saveTrafficHelper(incomingTraffic, writer);
            System.out.println("Incoming traffic saved to CSV successfully.");
        } catch (IOException e) {
            throw new RuntimeException("[" + new Date() + "] Error saving outgoing traffic to CSV: " + e.getMessage(), e);
        }
    }

    /**
     * Saves traffic data to a CSV file in a specific format.
     *
     * @param trafficList The list of traffic records to save.
     * @param writer      The writer used to save the data to a file.
     * @throws IOException If an I/O error occurs while writing the file.
     */
    private void saveTrafficHelper(List<EdgeTraffic> trafficList, BufferedWriter writer) throws IOException {
        StringBuilder dateRow = new StringBuilder("Date");
        for (EdgeTraffic traffic : trafficList) {
            dateRow.append(",").append(traffic.getDate());  // Append each date
        }
        writer.write(dateRow.toString());
        writer.newLine();  // Move to the next line after writing the dates

        StringBuilder trafficRow = new StringBuilder("Traffic");
        for (EdgeTraffic traffic : trafficList) {
            trafficRow.append(",").append(traffic.getTraffic());  // Append each traffic value
        }
        writer.write(trafficRow.toString());
        writer.newLine();  // Move to the next line after writing the traffic values
    }

    /**
     * Creates an empty JSON file for storing outgoing traffic data.
     */
    private void createOutgoingTrafficFile() {
        try {
            String emptyTrafficJson = gson.toJson(Collections.emptyList());
            Files.writeString(Paths.get(OUTGOING_TRAFFIC_FILE), emptyTrafficJson, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("[" + new Date() + "] Error creating outgoing traffic file: " + e.getMessage(), e);
        }
    }

    /**
     * Creates an empty JSON file for storing incoming traffic data.
     */
    private void createIncomingTrafficFile() {
        try {
            String emptyTrafficJson = gson.toJson(Collections.emptyList());
            Files.writeString(Paths.get(INCOMING_TRAFFIC_FILE), emptyTrafficJson);
        } catch (IOException e) {
            throw new RuntimeException("[" + new Date() + "] Error creating incoming traffic file: " + e.getMessage(), e);
        }
    }

    /**
     * Returns a new Gson instance for JSON serialization and deserialization.
     *
     * @return A Gson instance.
     */
    private Gson getGson() {
        return new Gson();
    }

    /**
     * Updates the traffic data by merging duplicate entries based on date.
     *
     * @param oldTraffic The list of traffic records to update.
     */
    private void updateHelper(List<EdgeTraffic> oldTraffic) {
        List<EdgeTraffic> newTraffic = new ArrayList<>();
        oldTraffic.forEach(edgeTraffic -> {
            if (!isDateContainedInTraffic(newTraffic, edgeTraffic.getDate())) {
                newTraffic.add(edgeTraffic);
            } else {
                EdgeTraffic presentTraffic = getEdgeTrafficByDate(newTraffic, edgeTraffic.getDate());
                newTraffic.remove(presentTraffic);
                presentTraffic.setTraffic(presentTraffic.getTraffic() + edgeTraffic.getTraffic());
                newTraffic.add(presentTraffic);
            }
        });
        oldTraffic.clear();
        oldTraffic.addAll(newTraffic);
    }
}