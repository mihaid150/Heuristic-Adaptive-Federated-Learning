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
import java.util.*;
import java.util.concurrent.CompletableFuture;

/**
 * A utility class providing various services for cloud nodes, such as notifying fog nodes,
 * managing edge-to-fog associations, broadcasting models, and clearing cache files.
 */
@Component
@RequiredArgsConstructor
public class CloudServiceUtils {
    private final DeviceManager deviceManager;
    private final PathManager pathManager;
    @Getter
    private final HttpHeaders jsonHeaders = createJsonHeaders();

    /**
     * Creates HTTP headers for json file transmission.
     *
     * @return HttpHeaders configured for json file data communication.
     */
    private HttpHeaders createJsonHeaders() {
        HttpHeaders headers = new org.springframework.http.HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);
        return headers;
    }

    /**
     * Notifies all fog nodes about the cloud host for communication purposes.
     *
     * @param restTemplate The {@link RestTemplate} used to send HTTP requests.
     */
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

    /**
     * Associates edges to fog nodes and informs each fog node about its associated edges.
     *
     * @param restTemplate The {@link RestTemplate} used to send HTTP requests.
     */
    public void associateEdgesToFogProcess(RestTemplate restTemplate) {
        // informs each fog what edges are associated with it

        // converts the map with edges to a queue to better usage
        Queue<EdgeTuple> edgesQueue = new LinkedList<>();
        for (Map.Entry<String, List<String>> entry : deviceManager.getEdgesMap().entrySet()) {
            edgesQueue.add(new EdgeTuple(entry.getValue().get(0), entry.getKey(), entry.getValue().get(1)));
        }

        // for each fog create a mini stack with its edges and send to each the notification
        for (Map.Entry<String, String> entry : deviceManager.getFogsMap().entrySet()) {
            Stack<EdgeTuple> edgeTupleStack = new Stack<>();
            List<EdgeTuple> edgeTupleList = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                EdgeTuple edgeTuple = edgesQueue.remove();
                edgeTupleStack.push(edgeTuple);
                edgeTupleList.add(edgeTuple);
            }
            deviceManager.getAssociatedEdgesToFogMap().put(entry.getKey(), edgeTupleList);
            notifyEdgesFromFog(edgeTupleStack, entry.getKey(), restTemplate);
        }
    }

    /**
     * Sends notifications to edges from a specific fog node.
     *
     * @param edgeTupleStack The stack of {@link EdgeTuple} objects representing the edges associated with the fog node.
     * @param fogHost        The host address of the fog node.
     * @param restTemplate   The {@link RestTemplate} used to send HTTP requests.
     */
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

    /**
     * Sends an HTTP POST request to a specified URL with the given body and headers.
     *
     * @param url           The URL to send the POST request to.
     * @param body          A {@link MultiValueMap} containing the body of the request.
     * @param headers       The {@link HttpHeaders} to include in the request.
     * @param restTemplate  The {@link RestTemplate} used to send the HTTP request.
     */
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

    /**
     * Broadcasts a model file to all fog nodes with specific training configurations.
     *
     * @param resource                 The {@link FileSystemResource} representing the model file to broadcast.
     * @param dates                    A list of dates representing the training timeline.
     * @param restTemplate             The {@link RestTemplate} used to send HTTP requests.
     * @param formHeaders              The {@link HttpHeaders} for the HTTP request.
     * @param concurrencyManager       The {@link ConcurrencyManager} for managing asynchronous tasks.
     * @param isCacheActive            A boolean indicating if caching is active.
     * @param isInitialTraining        A boolean indicating if this is the initial training process.
     * @param aggregationType          The type of aggregation to perform.
     * @param geneticEvaluationStrategy The strategy for genetic evaluation, if applicable.
     */
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

    /**
     * Sends a broadcast request to a specific fog node.
     *
     * @param body           The {@link MultiValueMap} containing the broadcast request body.
     * @param headers        The {@link HttpHeaders} for the HTTP request.
     * @param element        A map entry representing the fog node's IP address and name.
     * @param restTemplate   The {@link RestTemplate} used to send HTTP requests.
     * @param date           The date associated with the broadcast.
     * @param aggregationType The type of aggregation to perform.
     */
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

    /**
     * Broadcasts a model file to all edge nodes for evaluation.
     *
     * @param resource The {@link FileSystemResource} representing the model file to broadcast.
     * @param date     The date associated with the broadcast.
     */
    public void broadcastEvaluation(FileSystemResource resource, String date) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        RestTemplate restTemplate = new RestTemplate();



        Map<String, List<String>> edgesMap = deviceManager.getEdgesMap();
        if (edgesMap != null) {
            edgesMap.forEach((edgeHost, edgeInfo) -> {
                        String url = "http://192.168.2." + edgeHost + ":8080/edge/evaluate-model";

                        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
                        body.add("file", resource);
                        body.add("date", date);
                        body.add("lclid", edgeInfo.get(1));
                        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

                        ResponseEntity<String> response = restTemplate.postForEntity(url, requestEntity, String.class);
                        if (response.getStatusCode().is2xxSuccessful()) {
                            System.out.println(date + ": Successfully evaluation done one : " + edgeInfo.get(0));
                        } else {
                            System.err.println("Failed to evaluate on : " + edgeInfo.get(0) + ". Status code: " +
                                    response.getStatusCode());
                        }
                    }
            );
        }
    }

    /**
     * Creates a multi-part body for broadcasting a model to fog nodes.
     *
     * @param resource                 The {@link FileSystemResource} representing the model file to broadcast.
     * @param dates                    A list of dates associated with the training timeline.
     * @param isInitialTraining        A boolean indicating if this is the initial training process.
     * @param isCacheActive            A boolean indicating if caching is active.
     * @param aggregationType          The type of aggregation to perform.
     * @param geneticEvaluationStrategy The strategy for genetic evaluation, if applicable.
     * @return A {@link MultiValueMap} containing the body data for the broadcast request.
     */
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

    /**
     * Deletes all JSON files acting as cache in the specified folder.
     */
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

    /**
     * Deletes all fog model files stored in the cloud directory.
     */
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

    /**
     * Deletes the results file from the cloud directory.
     */
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
