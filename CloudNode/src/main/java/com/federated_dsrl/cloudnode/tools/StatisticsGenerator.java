package com.federated_dsrl.cloudnode.tools;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.federated_dsrl.cloudnode.entity.CloudTraffic;
import com.federated_dsrl.cloudnode.config.DeviceManager;
import com.federated_dsrl.cloudnode.config.PathManager;
import com.federated_dsrl.cloudnode.entity.EdgePerformanceResult;
import com.federated_dsrl.cloudnode.entity.ElapsedTimeFog;
import lombok.RequiredArgsConstructor;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.stream.Collectors;


@Component
@RequiredArgsConstructor
public class StatisticsGenerator {
    private final ConcurrencyManager concurrencyManager;
    private final PathManager pathManager;
    private final DeviceManager deviceManager;
    private final CloudTraffic cloudTraffic;

    public ResponseEntity<?> createElapsedTimeChart() throws IOException, InterruptedException {
        // load the json containing recorded elapsed time
        concurrencyManager.loadCacheFromJsonFile();

        // filter for the initial iteration
        // sort the elapsed time after dates
        // format them in a map
        Map<String, Double> filteredElapsedTimeIterations = concurrencyManager.getElapsedTimeOverIterations()
                .entrySet()
                .stream()
                .filter(entry -> !entry.getKey().contains("2013-07-09"))
                .sorted(Map.Entry.comparingByKey((key1, key2) -> {
                    // strip unwanted characters before comparing the dates
                    String cleanedKey1 = key1.replaceAll("[\"\\[\\]]", "").trim();
                    String cleanedKey2 = key2.replaceAll("[\"\\[\\]]", "").trim();
                    return cleanedKey1.compareTo(cleanedKey2);
                }))
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (oldValue, newValue) -> oldValue,
                        LinkedHashMap::new  // to preserve the order of insertion
                ));


        // convert map to comma-separated strings for saving in a CSV later on
        String elapsedTimes = filteredElapsedTimeIterations.values().stream()
                .map(Object::toString)
                .collect(Collectors.joining(","));
        String dates = String.join(",", filteredElapsedTimeIterations.keySet());

        // load the script which will generate the chart
        File scriptFile = new File(pathManager.getCloudStatisticsScriptPath());
        List<String> command = new ArrayList<>();
        command.add(pathManager.getPython3ExecutablePath());
        command.add(scriptFile.getAbsolutePath());
        command.add(elapsedTimes);
        command.add(dates);


        // execute the script
        processExecutor(command);
        return ResponseEntity.ok("Created successfully the chart!");
    }

    public ResponseEntity<?> createElapsedTimeChartsFogLayer(RestTemplate restTemplate) {
        // collect the elapsed time from the fogs by requests
        Map<String, ElapsedTimeFog[]> elapsedTimeFogMap = new HashMap<>();
        deviceManager.getFogsMap().forEach((fogHost, fogName) -> {
            String url = "http://192.168.2." + fogHost + ":8080/fog/request-elapsed-time-list";
            ResponseEntity<ElapsedTimeFog[]> response = restTemplate.getForEntity(url, ElapsedTimeFog[].class);

            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                elapsedTimeFogMap.put(fogName, response.getBody());
            }
        });

        // format, filter the elapsed time lists
        Map<String, ElapsedTimeFog[]> filteredElapsedTimeFogMap = elapsedTimeFogMap.entrySet().stream()
                .filter(entry -> !entry.getKey().contains("2013-07-09"))
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (oldValue, newValue) -> oldValue,
                        LinkedHashMap::new  // Preserve the order of insertion
                ));

        // convert the map to jSON
        ObjectMapper objectMapper = new ObjectMapper();
        String jsonData;
        try {
            jsonData = objectMapper.writeValueAsString(filteredElapsedTimeFogMap);
        } catch (IOException e) {
            return ResponseEntity.status(500).body("Failed to convert elapsed time data to JSON.");
        }

        // load the file to create the chart with elapsed time for every fog
        File scriptFile = new File(pathManager.getFogStatisticsScriptPath());
        List<String> command = List.of(pathManager.getPython3ExecutablePath(), scriptFile.getAbsolutePath(), jsonData);

        try {
            int exitCode = processExecutor(command);
            if (exitCode != 0) {
                return ResponseEntity.status(500).body("Failed to execute python script statistics.");
            }
        } catch (IOException | InterruptedException e) {
            return ResponseEntity.status(500).body("Error running Python script: " + e.getMessage());
        }
        return ResponseEntity.ok("Elapsed time charts for fog layer created successfully!");
    }

    public ResponseEntity<?> createTrafficChart(RestTemplate restTemplate) {
        // load the traffic from the json where it was stored
        cloudTraffic.loadTrafficFromJsonFile();

        // get the current iteration registered traffic
        List<Double> cloudIncomingTraffic = cloudTraffic.getIncomingTrafficOverIterations();
        List<Double> cloudOutgoingTraffic = cloudTraffic.getOutgoingTrafficOverIterations();

        // we do not want to take into account the first value(iteration)
        cloudIncomingTraffic.remove(0);
        cloudOutgoingTraffic.remove(0);

        // doing the requests on fogs and edges to get their registered traffic
        // first list for every fog and second for every iteration
        List<List<Double>> fogIncomingTrafficList = new ArrayList<>();
        List<List<Double>> fogOutgoingTrafficList = new ArrayList<>();
        Map<String, List<List<Double>>> edgeIncomingTrafficListMap = new HashMap<>(); // where String is for parent fog
        Map<String, List<List<Double>>> edgeOutGoingTrafficListMap = new HashMap<>();
        deviceManager.getFogsMap().forEach((fogHost, fogName) -> {
            String urlIncoming = "http://192.168.2." + fogHost + ":8080/fog/request-incoming-fog-traffic";
            String urlOutgoing = "http://192.168.2." + fogHost + ":8080/fog/request-outgoing-fog-traffic";
            ResponseEntity<Double[]> responseIncoming = restTemplate.getForEntity(urlIncoming, Double[].class);
            ResponseEntity<Double[]> responseOutgoing = restTemplate.getForEntity(urlOutgoing, Double[].class);
            if (responseIncoming.getStatusCode().is2xxSuccessful() && responseOutgoing.getStatusCode().is2xxSuccessful()
            && responseOutgoing.getBody() != null && responseIncoming.getBody() != null) {
                List<Double> temp = new ArrayList<>(List.of(responseIncoming.getBody()));
                temp.remove(0);
                fogIncomingTrafficList.add(temp);
                temp = new ArrayList<>(List.of(responseOutgoing.getBody()));
                temp.remove(0);
                fogOutgoingTrafficList.add(temp);
            }
            List<List<Double>> edgeIncomingTrafficList = new ArrayList<>();
            List<List<Double>> edgeOutgoingTrafficList = new ArrayList<>();
            deviceManager.getAssociatedEdgesToFogMap().get(fogHost).forEach(edge -> {
                String edgeUrlIncoming = "http://192.168.2." + edge.getHost() +
                        ":8080/edge/request-incoming-edge-traffic";
                String edgeUrlOutgoing = "http://192.168.2." + edge.getHost() +
                        ":8080/edge/request-outgoing-edge-traffic";

                ResponseEntity<List<Double>> edgeResponseIncoming = restTemplate.exchange(
                        edgeUrlIncoming,
                        HttpMethod.GET,
                        null,
                        new ParameterizedTypeReference<>() {
                        }
                );

                ResponseEntity<List<Double>> edgeResponseOutgoing = restTemplate.exchange(
                        edgeUrlOutgoing,
                        HttpMethod.GET,
                        null,
                        new ParameterizedTypeReference<>() {
                        }
                );

                if (edgeResponseIncoming.getStatusCode().is2xxSuccessful() && edgeResponseOutgoing.getStatusCode()
                        .is2xxSuccessful() && edgeResponseOutgoing.getBody() != null && edgeResponseIncoming.getBody()
                        != null) {
                    List<Double> incomingEdgeTrafficOverIterations = edgeResponseIncoming.getBody();
                    incomingEdgeTrafficOverIterations.remove(0);
                    List<Double> outgoingEdgeTrafficOverIterations = edgeResponseOutgoing.getBody();
                    outgoingEdgeTrafficOverIterations.remove(0);
                    edgeIncomingTrafficList.add(incomingEdgeTrafficOverIterations);
                    edgeOutgoingTrafficList.add(outgoingEdgeTrafficOverIterations);
                }
            });
            edgeIncomingTrafficListMap.put(fogName, edgeIncomingTrafficList);
            edgeOutGoingTrafficListMap.put(fogName, edgeOutgoingTrafficList);
        });

        // convert traffic data to json for saving it
        ObjectMapper objectMapper = new ObjectMapper();
        String cloudIncomingTrafficJson;
        String cloudOutgoingTrafficJson;
        String fogIncomingTrafficListJson;
        String fogOutgoingTrafficListJson;
        String edgeIncomingTrafficListMapJson;
        String edgeOutGoingTrafficListMapJson;

        try {
            cloudIncomingTrafficJson = objectMapper.writeValueAsString(cloudIncomingTraffic);
            cloudOutgoingTrafficJson = objectMapper.writeValueAsString(cloudOutgoingTraffic);
            fogIncomingTrafficListJson = objectMapper.writeValueAsString(fogIncomingTrafficList);
            fogOutgoingTrafficListJson = objectMapper.writeValueAsString(fogOutgoingTrafficList);
            edgeIncomingTrafficListMapJson = objectMapper.writeValueAsString(edgeIncomingTrafficListMap);
            edgeOutGoingTrafficListMapJson = objectMapper.writeValueAsString(edgeOutGoingTrafficListMap);
        } catch (IOException e) {
            return ResponseEntity.status(500).body("Failed to convert traffic data to JSON.");
        }

        // call the executable file for chart generation and prepare the arguments
        File scriptFile = new File(pathManager.getTrafficStatisticsScriptPath());
        List<String> command = List.of(
                pathManager.getPython3ExecutablePath(),
                scriptFile.getAbsolutePath(),
                cloudIncomingTrafficJson,
                cloudOutgoingTrafficJson,
                fogIncomingTrafficListJson,
                fogOutgoingTrafficListJson,
                edgeIncomingTrafficListMapJson,
                edgeOutGoingTrafficListMapJson
        );

        try {
            int exitCode = processExecutor(command);
            if (exitCode != 0) {
                return ResponseEntity.status(500).body("Failed to execute python script statistics.");
            }
        } catch (IOException | InterruptedException e) {
            return ResponseEntity.status(500).body("Error running Python traffic script: " + e.getMessage());
        }

        return ResponseEntity.ok("Successfully created traffic chart!");
    }

    public ResponseEntity<?> createPerformanceChart(RestTemplate restTemplate) {
        Map<String, List<EdgePerformanceResult>> edgePerformanceMap = new HashMap<>();

        // get the performance registered by each edge node
        deviceManager.getEdgesMap().forEach((edgeHost, edgeInfo) -> {
            String url = "http://192.168.2." + edgeHost + ":8080/edge/request-performance-result";
            ResponseEntity<EdgePerformanceResult[]> response = restTemplate.getForEntity(url, EdgePerformanceResult[].class);
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                edgePerformanceMap.put(edgeInfo.get(0), List.of(response.getBody()));
            }
        });

        // convert the edgePerformanceMap to json for saving it
        ObjectMapper objectMapper = new ObjectMapper();
        String edgePerformanceJson;
        try {
            edgePerformanceJson = objectMapper.writeValueAsString(edgePerformanceMap);
        } catch (IOException e) {
            return ResponseEntity.status(500).body("Failed to convert edge performance data to JSON.");
        }

        // run the python script as executable
        File scriptFile = new File(pathManager.getPerformanceStatisticsScriptPath());
        List<String> command = List.of(pathManager.getPython3ExecutablePath(), scriptFile.getAbsolutePath(), edgePerformanceJson);

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        try {
            Process process = processBuilder.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }
            System.out.println("Python performance statistics script output: " + output);

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
                StringBuilder errorOutput = new StringBuilder();
                while ((line = errorReader.readLine()) != null) {
                    errorOutput.append(line).append("\n");
                }
                System.err.println("Error output from python performance statistics script: " + errorOutput);
                return ResponseEntity.status(500).body("Failed to execute python performance script.");
            }
        } catch (IOException | InterruptedException e) {
            return ResponseEntity.status(500).body("Error running Python performance script: " + e.getMessage());
        }

        return ResponseEntity.ok("Successfully created performance chart!");
    }

    private int processExecutor(List<String> command) throws IOException, InterruptedException {
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        Process process = processBuilder.start();
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        StringBuilder output = new StringBuilder();
        processBuilder.redirectErrorStream(true);
        String line;
        while ((line = reader.readLine()) != null) {
            output.append(line).append("\n");
        }
        System.out.println("Python statistics script output: " + output);

        int exitCode = process.waitFor();

        if (exitCode != 0) {
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            StringBuilder errorOutput = new StringBuilder();
            while ((line = errorReader.readLine()) != null) {
                errorOutput.append(line).append("\n");
            }
            System.err.println("Error output from python statistics script: " + errorOutput);
        }
        return exitCode;
    }
}
