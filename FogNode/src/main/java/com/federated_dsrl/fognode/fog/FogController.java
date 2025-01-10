package com.federated_dsrl.fognode.fog;

import com.federated_dsrl.fognode.config.FogEndpoints;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

/**
 * REST Controller for managing fog node operations in the federated learning framework.
 * <p>
 * This controller handles requests related to edge device association, cloud and edge model processing,
 * traffic monitoring, and readiness signaling.
 * </p>
 */
@RequestMapping(FogEndpoints.FOG_MAPPING)
@RestController
@RequiredArgsConstructor
public class FogController {

    private final FogService fogService;

    /**
     * Associates an edge device with the fog node.
     *
     * @param name  the name of the edge device
     * @param host  the host address of the edge device
     * @param lclid the local client ID (lclid) of the edge device
     * @return a confirmation message
     */
    @PostMapping(FogEndpoints.ADD_EDGE)
    public ResponseEntity<?> associateEdgeToFog(@RequestParam("name") String name, @RequestParam("host") String host,
                                                @RequestParam("lclid") String lclid) {
        fogService.addEdge(name, host, lclid);
        return ResponseEntity.ok("Associated edge " + name + " (host: " + host + ", lclid: " + lclid + ") to the fog.");
    }

    /**
     * Acknowledges the cloud node in the federated learning system.
     *
     * @param host the host address of the cloud
     * @param name the name of the cloud
     * @return a confirmation message
     */
    @PostMapping(FogEndpoints.ACK_CLOUD)
    public ResponseEntity<?> acknowledgeCloud(@RequestParam("host") String host, @RequestParam("name") String name) {
        fogService.ackCloud(host, name);
        return ResponseEntity.ok("The cloud with host " + host + " has been acknowledged.");
    }

    /**
     * Notifies edges about the current model type.
     *
     * @param modelType the type of model (e.g., LSTM)
     * @return a confirmation message
     */
    @PostMapping(FogEndpoints.EDGE_MODEL_TYPE)
    public ResponseEntity<?> notifyEdgeAboutModelType(@RequestParam("model_type") String modelType) {
        fogService.notifyEdgeAboutModelType(modelType);
        return ResponseEntity.ok("Successfully notified edges about model type: " + modelType);
    }

    /**
     * Receives the global model from the cloud and distributes it to associated edges.
     *
     * @param file                     the global model file
     * @param date                     the iteration date
     * @param isInitialTraining        whether this is the initial training
     * @param isCacheActive            whether caching is active
     * @param geneticEvaluationStrategy the genetic evaluation strategy to be used
     * @return a confirmation or error message
     */
    @PostMapping(FogEndpoints.RECEIVE_CLOUD_MODEL)
    public ResponseEntity<?> receiveGlobalModel(@RequestParam("file") MultipartFile file,
                                                @RequestParam("date") List<String> date,
                                                @RequestParam("is_initial_training") Boolean isInitialTraining,
                                                @RequestParam("is_cache_active") Boolean isCacheActive,
                                                @RequestParam("genetic_evaluation_strategy") String geneticEvaluationStrategy) {
        try {
            fogService.receiveCloudModel(file, date, isInitialTraining, isCacheActive, geneticEvaluationStrategy);
            return ResponseEntity.ok("Successfully sent cloud models to edges associated to this fog!");
        } catch (IOException e) {
            System.out.println("An exception occurred in the receive cloud model from fog controller: " + e);
            return ResponseEntity.badRequest().body("Bad request sent cloud models to edges associated to this fog!");
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Receives the retrained model from an edge device.
     *
     * @param newEdgePerformance the performance of the new edge model
     * @param oldFogPerformance  the performance of the old fog model
     * @param lclid              the local client ID (lclid) of the edge
     * @param edgeModel          the edge model file
     * @return a confirmation or error message
     */
    @PostMapping(FogEndpoints.RECEIVE_EDGE_MODEL)
    public ResponseEntity<?> receiveEdgeModel(@RequestParam("new_edge_performance") Double newEdgePerformance,
                                              @RequestParam("old_fog_performance") Double oldFogPerformance,
                                              @RequestParam("lclid") String lclid,
                                              @RequestParam("edge_model") MultipartFile edgeModel) {
        try {
            fogService.receiveEdgeModel(newEdgePerformance, oldFogPerformance, lclid, edgeModel);
            return ResponseEntity.ok("Received successfully the edge model from edge!");
        } catch (IOException e) {
            System.out.println("An exception occurred in the receiveEdgeModel from fog controller: " + e);
            return ResponseEntity.badRequest().body("Error receiving the edge model.");
        }
    }

    /**
     * Retrieves the fog model for the current iteration.
     *
     * @return the fog model response
     */
    @GetMapping(FogEndpoints.REQUEST_FOG_MODEL)
    public ResponseEntity<?> requestFogModel() {
        return fogService.requestFogModel();
    }

    /**
     * Receives a readiness signal from an edge device.
     *
     * @param lclid the local client ID (lclid) of the edge
     * @return a confirmation message
     */
    @PostMapping(FogEndpoints.RECEIVE_READINESS_SIGNAL)
    public ResponseEntity<?> receiveEdgeReadinessSignal(@RequestParam("lclid") String lclid) {
        fogService.receiveEdgeReadinessSignal(lclid);
        return ResponseEntity.ok("The ready signal from " + lclid + " edge was received!");
    }

    /**
     * Determines whether the specified edge should proceed with its operation.
     *
     * @param lclid the local client ID (lclid) of the edge
     * @return a response indicating if the edge should proceed
     */
    @GetMapping(FogEndpoints.EDGE_SHOULD_PROCEED)
    public ResponseEntity<?> edgeShouldProceed(@PathVariable String lclid) {
        return fogService.shouldEdgeProceed(lclid);
    }

    /**
     * Retrieves the elapsed time list for the fog node operations.
     *
     * @return the elapsed time list response
     */
    @GetMapping(FogEndpoints.REQUEST_ELAPSED_TIME_LIST)
    public ResponseEntity<?> requestElapsedTimeList() {
        return fogService.requestElapsedTimeList();
    }

    /**
     * Retrieves the incoming traffic data for the fog node.
     *
     * @return the incoming traffic response
     */
    @GetMapping(FogEndpoints.REQUEST_INCOMING_FOG_TRAFFIC)
    public ResponseEntity<?> requestIncomingFogTraffic() {
        return fogService.requestIncomingFogTraffic();
    }

    /**
     * Retrieves the outgoing traffic data for the fog node.
     *
     * @return the outgoing traffic response
     */
    @GetMapping(FogEndpoints.REQUEST_OUTGOING_FOG_TRAFFIC)
    public ResponseEntity<?> requestOutgoingFogTraffic() {
        return fogService.requestOutgoingFogTraffic();
    }

    /**
     * Loads the system state of the fog node.
     *
     * @return the system state response
     */
    @PostMapping(FogEndpoints.LOAD_SYSTEM_STATE)
    public ResponseEntity<?> loadSystemState() {
        return fogService.loadSystemState();
    }
}
