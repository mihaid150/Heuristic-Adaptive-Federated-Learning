package com.federated_dsrl.fognode.fog;

import com.federated_dsrl.fognode.config.FogEndpoints;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

/**
 * REST Controller for managing fog node operations using the FedAvg (Federated Averaging) approach
 * in the federated learning framework.
 * <p>
 * This controller handles requests related to cloud model reception, edge readiness signaling,
 * edge model reception, and determining edge progression for training operations.
 * </p>
 */
@RequestMapping(FogEndpoints.FOG_FAVG_MAPPING)
@RestController
@RequiredArgsConstructor
public class FogFAVGController {

    private final FogFAVGService fogFAVGService;

    /**
     * Receives the global (cloud) model and distributes it to the associated edges.
     *
     * @param file              the global model file received from the cloud
     * @param date              the iteration date
     * @param isInitialTraining whether this is the initial training phase
     * @return a response indicating the result of the operation
     */
    @PostMapping(FogEndpoints.RECEIVE_CLOUD_MODEL)
    public ResponseEntity<?> receiveCloudModel(@RequestParam("file") MultipartFile file,
                                               @RequestParam("date") List<String> date,
                                               @RequestParam("is_initial_training") Boolean isInitialTraining) {
        return fogFAVGService.receiveCloudModel(file, date, isInitialTraining);
    }

    /**
     * Receives a readiness signal from an edge device, indicating it is prepared for the next operation.
     *
     * @param lclid the local client ID (lclid) of the edge device
     * @return a confirmation message indicating that the readiness signal was received
     */
    @PostMapping(FogEndpoints.RECEIVE_READINESS_SIGNAL)
    public ResponseEntity<?> receiveEdgeReadinessSignal(@RequestParam("lclid") String lclid) {
        fogFAVGService.receiveEdgeReadinessSignal(lclid);
        return ResponseEntity.ok("The ready signal from " + lclid + " edge was received!");
    }

    /**
     * Determines whether the specified edge device should proceed with its next operation.
     *
     * @param lclid the local client ID (lclid) of the edge device
     * @return a response indicating whether the edge should proceed
     */
    @GetMapping(FogEndpoints.EDGE_SHOULD_PROCEED)
    public ResponseEntity<?> edgeShouldProceed(@PathVariable String lclid) {
        return fogFAVGService.shouldEdgeProceed(lclid);
    }

    /**
     * Receives the model from an edge device.
     *
     * @param newEdgePerformance the performance of the new edge model
     * @param oldFogPerformance  the performance of the old fog model
     * @param lclid              the local client ID (lclid) of the edge device
     * @param edgeModel          the edge model file received from the edge device
     * @return a response indicating the result of the operation
     */
    @PostMapping(FogEndpoints.RECEIVE_EDGE_MODEL)
    public ResponseEntity<?> receiveEdgeModel(@RequestParam("new_edge_performance") Double newEdgePerformance,
                                              @RequestParam("old_fog_performance") Double oldFogPerformance,
                                              @RequestParam("lclid") String lclid,
                                              @RequestParam("edge_model") MultipartFile edgeModel) {
        try {
            fogFAVGService.receiveEdgeModel(newEdgePerformance, oldFogPerformance, lclid, edgeModel);
            return ResponseEntity.ok("Received successfully the edge model from edge!");
        } catch (IOException e) {
            System.out.println("An exception occurred in the receiveEdgeModel from fog controller: " + e);
            return ResponseEntity.badRequest().body("Error occurred while receiving the edge model.");
        }
    }
}
