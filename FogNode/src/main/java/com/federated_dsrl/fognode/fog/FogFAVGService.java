package com.federated_dsrl.fognode.fog;

import com.federated_dsrl.fognode.config.DeviceManager;
import com.federated_dsrl.fognode.entity.FogTraffic;
import com.federated_dsrl.fognode.tools.ConcurrencyManager;
import com.federated_dsrl.fognode.utils.ModelFileHandler;
import com.federated_dsrl.fognode.config.EdgeReadinessManager;
import com.federated_dsrl.fognode.entity.EdgeEntity;
import com.federated_dsrl.fognode.config.AggregationType;
import com.federated_dsrl.fognode.tools.CountDownLatchManager;
import com.federated_dsrl.fognode.utils.FogServiceUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Service class for handling fog operations in the FedAvg (Federated Averaging) aggregation strategy.
 * <p>
 * This class provides functionality to receive cloud models, handle edge readiness, process edge models,
 * and perform model aggregation using the average strategy.
 * </p>
 */
@Service
@RequiredArgsConstructor
public class FogFAVGService {
    private final ModelFileHandler modelFileHandler;
    private final ConcurrencyManager concurrencyManager;
    private final FogTraffic fogTraffic;
    private final FogServiceUtils fogServiceUtils;
    private final DeviceManager deviceManager;
    private final EdgeReadinessManager edgeReadinessManager;
    private Boolean isFirstAggregation = Boolean.TRUE;
    private final CountDownLatchManager latchManager;

    /**
     * Receives the global (cloud) model and distributes it to associated edges for retraining.
     *
     * @param file            the global model file received from the cloud
     * @param date            the training iteration date
     * @param isInitialTraining whether this is the first training iteration
     * @return a {@link ResponseEntity} confirming successful receipt or an error message
     */
    public ResponseEntity<?> receiveCloudModel(MultipartFile file, List<String> date, Boolean isInitialTraining) {
        try {
            // notify edges about the current training date
            fogServiceUtils.setEdgeWorkingDate(date);

            // clean the local storage for the initial training iteration
            if (isInitialTraining) {
                fogServiceUtils.deleteCacheJsonFolderContent();
                fogTraffic.clearTraffic();
            }

            // remove the edge models from the previous iteration
            fogServiceUtils.deleteEdgeModelFiles();

            // to not sent to cloud a not aggregated model
            concurrencyManager.setAlreadySentFogModel(Boolean.FALSE);

            // initialize the elapsed time measuring
            concurrencyManager.initCurrentElapsedTimeManager();

            // load statistics cache
            fogTraffic.loadTrafficFromJsonFile();
            concurrencyManager.loadElapsedTimeToJsonFile();

            // inform up until is all ok and received successfully the cloud model
            fogServiceUtils.logInfo("Received cloud model with date: " + date + ", is initial training: " +
                    isInitialTraining);


            // check for a non-empty received model (edge case) and then save it locally
            modelFileHandler.validateModelFile(file);
            modelFileHandler.saveFogModel(file);

            // current fog save current training date and whether is initial training
            fogServiceUtils.setTrainingDate(date, isInitialTraining);

            // prepare the edge readiness state for when the edges send back the trained models
            edgeReadinessManager.resetReadiness(deviceManager.getEdges());
            edgeReadinessManager.initializeReadiness(deviceManager.getEdges());

            // load the cloud model now as fog model
            Path fogModelPath = modelFileHandler.getFogModelPath();

            // create a barrier for waiting to all training edges to send back their models
            latchManager.resetLatch(3);
            fogServiceUtils.logInfo("Initial latch count: " + latchManager.getLatchCount());

            // broadcast the fog model to children and wait for the response model back
            for (EdgeEntity edge : deviceManager.getEdges()) {
                fogServiceUtils.sendModelToEdgeAsync(edge, fogModelPath, date, isInitialTraining,
                        AggregationType.AVERAGE);
                fogServiceUtils.waitForEdgeReadinessAsync(edge);
            }

            fogServiceUtils.logInfo("Waiting for all edges to send model and become ready...");
            latchManager.await();
            fogServiceUtils.logInfo("All edges are ready. Starting cooling process.");

            // when all edges signaled that they received the fog model, signal them back to start aggregation
            edgeReadinessManager.signalEdgesToProceed(deviceManager.getEdges());
            return ResponseEntity.ok("All edges were signaled to proceed with retraining the model.");
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Bad request sent in receive cloud mode:" + e.getMessage());
        }
    }

    /**
     * Updates the readiness state of the edge that sent a readiness signal.
     *
     * @param lclid the local client ID (lclid) of the edge
     */
    public void receiveEdgeReadinessSignal(String lclid) {
        // update the readiness state of the edge who signaled
        fogServiceUtils.receiveEdgeReadinessSignal(lclid);
    }

    /**
     * Determines whether an edge should proceed with retraining.
     *
     * @param lclid the local client ID (lclid) of the edge
     * @return a {@link ResponseEntity} indicating whether the edge should proceed
     */
    public ResponseEntity<?> shouldEdgeProceed(String lclid) {
        // response to the edge request if it should start aggregation
        return fogServiceUtils.shouldEdgeProceed(lclid);
    }

    /**
     * Processes an edge model received from an edge device.
     * <p>
     * If all required edge models have been received, the method performs model aggregation and
     * sends the aggregated model to the cloud node.
     * </p>
     *
     * @param newEdgePerformance the performance of the new edge model
     * @param oldFogPerformance  the performance of the old fog model
     * @param lclid              the local client ID (lclid) of the edge
     * @param edgeModel          the edge model file
     * @throws IOException if an error occurs during file operations
     */
    public void receiveEdgeModel(Double newEdgePerformance, Double oldFogPerformance, String lclid,
                                 MultipartFile edgeModel) throws IOException {
        // only one edge model trained process at a time
        concurrencyManager.lock();
        try {
            fogServiceUtils.logInfo("Receiving edge model. New Edge Performance: " + newEdgePerformance +
                    ", Old Fog Performance: " + oldFogPerformance + ", LCID: " + lclid);

            // save locally the received edge model
            Path edgeModelPath = modelFileHandler.saveEdgeModel(edgeModel, lclid, fogServiceUtils);
            fogServiceUtils.logInfo("New edge model was saved to: " + edgeModelPath);

            // only when there are 3 models(all edge models in this case) saved, start the aggregation
            if (modelFileHandler.countReceivedEdgeModels() == 3 && !concurrencyManager.getAlreadySentFogModel()) {
                aggregateModels();
                // to not send again to edges the fog model and they to train again and send back an unwanted model
                concurrencyManager.setAlreadySentFogModel(Boolean.TRUE);
                // when new aggregated fog model done, send it to the cloud node
                fogServiceUtils.sendFogModelToCloud(AggregationType.AVERAGE);

                // generate up until now statistics with the training and aggregation processes
                fogServiceUtils.statisticsHelper();
            }
        } finally {
            // let the other requests to be processed
            concurrencyManager.unlock();
        }
    }

    /**
     * Aggregates received edge models using the average strategy.
     */
    private void aggregateModels() {
        // load the script for aggregation and execute it
        System.out.println("Aggregating received edge models in average strategy.");
        List<String> command = modelFileHandler.prepareAverageAggregationCommand(isFirstAggregation);
        if (isFirstAggregation) {
            // set flag for initial aggregation
            this.isFirstAggregation = Boolean.FALSE;
        }
        // second parameters for type of aggregation in FAVG case
        modelFileHandler.runScript(command, "average");
    }
}
