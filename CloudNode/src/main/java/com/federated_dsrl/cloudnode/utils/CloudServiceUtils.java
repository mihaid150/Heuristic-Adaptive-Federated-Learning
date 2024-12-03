package com.federated_dsrl.cloudnode.utils;

import com.federated_dsrl.cloudnode.config.GeneticEvaluationStrategy;
import com.federated_dsrl.cloudnode.entity.EdgeTuple;
import com.federated_dsrl.cloudnode.tools.ConcurrencyManager;
import com.federated_dsrl.cloudnode.config.AggregationType;
import com.federated_dsrl.cloudnode.config.DeviceManager;
import com.federated_dsrl.cloudnode.config.PathManager;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import java.util.concurrent.CompletableFuture;


@Component
@RequiredArgsConstructor
public class CloudServiceUtils {
    private final DeviceManager deviceManager;
    private final PathManager pathManager;
    @Getter
    private final HttpHeaders jsonHeaders = createJsonHeaders();

    private HttpHeaders createJsonHeaders() {
        HttpHeaders headers = new org.springframework.http.HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);
        return headers;
    }

    public void notifyFogAboutCloud(RestTemplate restTemplate) {

        // notifies all the fog children nodes about the cloud IP/HOST for communication purpose
        String hostAddress = System.getenv("HOST_IP");
        System.out.println("Cloud IP address: " + hostAddress);
        String[] parts = hostAddress.split("\\.");
        String lastByte = parts[parts.length - 1];

        for (Map.Entry<String, String> entry : deviceManager.getFogsMap().entrySet()) {
            MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
            body.add("host", lastByte);
            body.add("name", entry.getValue());
            HttpEntity<MultiValueMap<String, String>> requestEntity = new HttpEntity<>(body, jsonHeaders);
            ResponseEntity<String> response = restTemplate.postForEntity("http://192.168.2." + entry.getKey() + ":8080/fog/ack-cloud", requestEntity, String.class);
            if (response.getStatusCode().is2xxSuccessful()) {
                System.out.println("Successfully notified fog " + entry.getValue() + " about cloud host.");
            } else {
                System.err.println("Unsuccessfully notified fog " + entry.getValue() + " about cloud host." + ". " + "Status code: " + response.getStatusCode());
            }
        }
    }

    public void associateEdgesToFogProcess(RestTemplate restTemplate) {
        // informs each fog what edges are associated with it

        // converts the map with edges to a stack to better usage
        Stack<EdgeTuple> edgesStack = new Stack<>();
        for (Map.Entry<String, List<String>> entry : deviceManager.getEdgesMap().entrySet()) {
            edgesStack.push(new EdgeTuple(entry.getValue().get(0), entry.getKey(), entry.getValue().get(1)));
        }

        // for each fog create a mini stack with its edges and send to each the notification
        for (Map.Entry<String, String> entry : deviceManager.getFogsMap().entrySet()) {
            Stack<EdgeTuple> edgeTupleStack = new Stack<>();
            List<EdgeTuple> edgeTupleList = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                EdgeTuple edgeTuple = edgesStack.pop();
                edgeTupleStack.push(edgeTuple);
                edgeTupleList.add(edgeTuple);
            }
            deviceManager.getAssociatedEdgesToFogMap().put(entry.getKey(), edgeTupleList);
            notifyEdgesFromFog(edgeTupleStack, entry.getKey(), restTemplate);
        }
    }

    private void notifyEdgesFromFog(Stack<EdgeTuple> edgeTupleStack, String fogHost, RestTemplate restTemplate) {
        while (!edgeTupleStack.isEmpty()) {
            EdgeTuple tuple = edgeTupleStack.pop();
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("name", tuple.getName());
            body.add("host", tuple.getHost());
            body.add("lclid", tuple.getMac());

            sendPostRequest("http://192.168.2." + fogHost + ":8080/fog/add-edge", body, jsonHeaders, restTemplate);
        }
    }

    private void sendPostRequest(String url, MultiValueMap<String, ?> body, HttpHeaders headers,
                                 RestTemplate restTemplate) {
        HttpEntity<MultiValueMap<String, ?>> requestEntity = new HttpEntity<>(body, headers);
        ResponseEntity<String> response = restTemplate.postForEntity(url, requestEntity, String.class);

        if (response.getStatusCode().is2xxSuccessful()) {
            System.out.println("Request to " + url + " was successful.");
        } else {
            System.err.println("Request to " + url + " failed. Status code: " + response.getStatusCode());
        }
    }

    public void broadcast(FileSystemResource resource, List<String> dates, RestTemplate restTemplate, HttpHeaders
            formHeaders, ConcurrencyManager concurrencyManager, Boolean isCacheActive, Boolean isInitialTraining,
                          AggregationType aggregationType, GeneticEvaluationStrategy geneticEvaluationStrategy) {
        List<CompletableFuture<Void>> futures = new ArrayList<>();
        String date = dates.size() == 2 ? dates.get(1) : dates.get(0);

        for (Map.Entry<String, String> element : deviceManager.getFogsMap().entrySet()) {
            CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                try {
                    MultiValueMap<String, Object> body = createBroadcastBody(resource, dates, isInitialTraining,
                            isCacheActive, aggregationType, geneticEvaluationStrategy);
                    broadcastHelper(body, formHeaders, element, restTemplate, date, aggregationType);
                } catch (Exception e) {
                    System.err.println("Exception occurred while sending model to: " + element.getValue() + ". " +
                            "Error: " + e.getMessage());
                }
            }, concurrencyManager.getExecutorService());
            futures.add(future);
        }
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
    }

    private void broadcastHelper(MultiValueMap<String, Object> body, HttpHeaders headers,
                                 Map.Entry<String, String> element, RestTemplate restTemplate, String date,
                                 AggregationType aggregationType) {
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
        String url = aggregationType.equals(AggregationType.GENETIC) ? "http://192.168.2." + element.getKey() +
                ":8080/fog/receive-global-model" : "http://192.168.2." + element.getKey() +
                ":8080/fog-favg/receive-global-model";
        ResponseEntity<String> response = restTemplate.postForEntity(url, requestEntity, String.class);

        if (response.getStatusCode().is2xxSuccessful()) {
            System.out.println(date + ": Successfully sent model to: " + element.getValue());
        } else {
            System.err.println("Failed to send model to: " + element.getValue() + ". Status code: " + response.getStatusCode());
        }
    }

    private MultiValueMap<String, Object> createBroadcastBody(FileSystemResource resource, List<String> dates,
                                                              boolean isInitialTraining, boolean isCacheActive,
                                                              AggregationType aggregationType,
                                                              GeneticEvaluationStrategy geneticEvaluationStrategy) {
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", resource);
        body.add("date", dates);
        body.add("is_initial_training", isInitialTraining);
        body.add("is_cache_active", isCacheActive);
        if (aggregationType.equals(AggregationType.GENETIC)) {
            body.add("genetic_evaluation_strategy", geneticEvaluationStrategy);
        }
        return body;
    }

    public void deleteCacheJsonFolderContent() {

        // it removes from the storage all the json files acting as cache
        // in the cache json files the cumulative lambda is stored TODO check before affirmation
        File directory = new File("/app/cache_json");
        if (!directory.exists() || !directory.isDirectory()) {
            System.err.println("Invalid directory, unable to delete fog model files.");
            return;
        }

        File[] cacheJsonFiles = directory.listFiles((dir, name) -> name.endsWith(".json"));
        if (cacheJsonFiles == null || cacheJsonFiles.length == 0) {
            System.err.println("No json files found to delete.");
            return;
        }

        for (File jsonFile : cacheJsonFiles) {
            if (jsonFile.delete()) {
                System.out.printf("Deleted json file: " + jsonFile.getName());
            } else {
                System.err.println("Failed to delete json file:" + jsonFile.getName());
            }
        }
    }

    public void deleteFogModelFiles() {
        File directory = new File(pathManager.getCloudDirectoryPath());
        if (!directory.exists() || !directory.isDirectory()) {
            System.err.println("Invalid directory, unable to delete fog model files.");
            return;
        }

        File[] fogModelFiles = directory.listFiles((dir, name) -> name.startsWith("fog_model") && name.endsWith(".keras"));
        if (fogModelFiles == null || fogModelFiles.length == 0) {
            System.out.println("No fog model files found to delete.");
            return;
        }

        for (File fogModelFile : fogModelFiles) {
            if (fogModelFile.delete()) {
                System.out.println("Deleted fog model file: " + fogModelFile.getName());
            } else {
                System.err.println("Failed to delete fog model file: " + fogModelFile.getName());
            }
        }
    }

    public void deleteResultFile() {

        // removes all files that store previous results
        File resultFile = new File(pathManager.getCloudDirectoryPath() + "/results.json");

        if (resultFile.delete()) {
            System.out.println("Deleted result file.");
        } else {
            System.out.println("No results file has been found.");
        }
    }
}
