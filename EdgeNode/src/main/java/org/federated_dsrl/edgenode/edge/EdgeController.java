package org.federated_dsrl.edgenode.edge;

import lombok.RequiredArgsConstructor;
import org.federated_dsrl.edgenode.config.EdgeEndpoints;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

/**
 * Controller for handling edge-related operations, including model training,
 * parameter setting, and traffic metrics.
 */
@RequestMapping(EdgeEndpoints.EDGE_MAPPING)
@RestController
@RequiredArgsConstructor
public class EdgeController {
    private final EdgeService edgeService;

    /**
     * Identifies the parent fog node for the edge.
     *
     * @param host The host address of the parent fog node.
     * @return A response acknowledging the parent fog.
     */
    @PostMapping(EdgeEndpoints.IDENTIFY_PARENT_FOG)
    public ResponseEntity<?> identifyParentFog(@RequestParam("host") String host) {
        System.out.println("host from edge: " + host);
        edgeService.identifyParentFog(host);
        return ResponseEntity.ok("Parent fog has been acknowledged.");
    }

    /**
     * Sets the model type to be used for training.
     *
     * @param model_type The type of the model (e.g., LSTM).
     * @return A response indicating the model type was set successfully.
     */
    @PostMapping(EdgeEndpoints.SET_MODEL_TYPE)
    public ResponseEntity<?> setModelType(@RequestParam("model_type") String model_type) {
        edgeService.setModelType(model_type);
        return ResponseEntity.ok("Model type was set to: " + model_type);
    }

    /**
     * Retrains the fog model with the provided data and configuration.
     *
     * @param file             The fog model file.
     * @param date             The training date(s).
     * @param lclid            The logical cluster ID.
     * @param initialTraining  Whether this is the initial training step.
     * @param aggregationType  The aggregation type used for training.
     * @return A response indicating the retraining process is in progress.
     * @throws IOException          If an error occurs during file handling.
     * @throws InterruptedException If the thread is interrupted during training.
     */
    @PostMapping(EdgeEndpoints.RECEIVE_FOG_MODEL)
    public ResponseEntity<?> retrainFogModel(@RequestParam("file") MultipartFile file,
                                             @RequestParam("date") List<String> date,
                                             @RequestParam("lclid") String lclid,
                                             @RequestParam("initial_training") Boolean initialTraining,
                                             @RequestParam("aggregation_type") String aggregationType)
            throws IOException, InterruptedException {
        if (initialTraining) {
            try {
                System.out.println("in the retrain fog model initialization with " + lclid);
                edgeService.initialization(date.get(0), date.get(1), lclid, aggregationType);
            } catch (IOException | InterruptedException e) {
                System.out.println("exception in initialization: " + e.getMessage());
                throw new RuntimeException(e);
            }
        } else {
            edgeService.retrainFogModel(file, date, lclid, aggregationType);
        }
        return ResponseEntity.ok("in working...");
    }

    /**
     * Performs genetics-based training with the provided parameters.
     *
     * @param file          The model file.
     * @param date          The training date(s).
     * @param lclid         The logical cluster ID.
     * @param learningRate  The learning rate for training.
     * @param batchSize     The batch size for training.
     * @param epochs        The number of epochs for training.
     * @param patience      The early stopping patience value.
     * @param fineTuning    The number of fine-tuning layers.
     * @return A response containing the results of the genetics training.
     */
    @PostMapping(EdgeEndpoints.GENETICS_TRAINING)
    public ResponseEntity<?> geneticsTraining(@RequestParam("file") MultipartFile file,
                                              @RequestParam("date") List<String> date,
                                              @RequestParam("lclid") String lclid,
                                              @RequestParam("learning_rate") String learningRate,
                                              @RequestParam("batch_size") String batchSize,
                                              @RequestParam("epochs") String epochs,
                                              @RequestParam("patience") String patience,
                                              @RequestParam("fine_tune_layers") String fineTuning) {
        try {
            return edgeService.doGeneticsValidation(file, date, lclid, learningRate, batchSize, epochs, patience,
                    fineTuning);
        } catch (IOException | InterruptedException e) {
            System.out.println("error from genetics training: " + e.getMessage());
            throw new RuntimeException(e);
        }
    }

    /**
     * Receives training parameters from the client.
     *
     * @param learningRate The learning rate for training.
     * @param batchSize    The batch size for training.
     * @param epochs       The number of epochs for training.
     * @param patience     The early stopping patience value.
     * @param fineTune     The number of fine-tuning layers.
     * @return A response confirming the training parameters were received.
     */
    @PostMapping(EdgeEndpoints.RECEIVE_PARAMS)
    public ResponseEntity<?> receiveTrainingParameters(@RequestParam("learning_rate") String learningRate,
                                                       @RequestParam("batch_size") String batchSize,
                                                       @RequestParam("epochs") String epochs,
                                                       @RequestParam("patience") String patience,
                                                       @RequestParam("fine_tune") String fineTune) {
        return edgeService.setTrainingParameters(learningRate, batchSize, epochs, patience, fineTune);
    }

    /**
     * Retrieves incoming edge traffic metrics.
     *
     * @return A list of incoming edge traffic metrics.
     */
    @GetMapping(EdgeEndpoints.REQUEST_INCOMING_EDGE_TRAFFIC)
    public ResponseEntity<List<Double>> requestIncomingEdgeTraffic() {
        return edgeService.requestIncomingEdgeTraffic();
    }

    /**
     * Retrieves outgoing edge traffic metrics.
     *
     * @return A list of outgoing edge traffic metrics.
     */
    @GetMapping(EdgeEndpoints.REQUEST_OUTGOING_EDGE_TRAFFIC)
    public ResponseEntity<List<Double>> requestOutgoingFogTraffic() {
        return edgeService.requestOutgoingEdgeTraffic();
    }

    /**
     * Requests the performance result of the current edge model.
     *
     * @return A response containing the performance result.
     */
    @GetMapping(EdgeEndpoints.REQUEST_PERFORMANCE_RESULT)
    public ResponseEntity<?> requestPerformanceResult() {
        return edgeService.requestPerformanceResult();
    }

    /**
     * Sets the working date for the edge node.
     *
     * @param date The date to set.
     * @return A response confirming the date was set.
     */
    @PostMapping(EdgeEndpoints.SET_WORKING_DATE)
    public ResponseEntity<?> setWorkingDate(@RequestParam("date") String date) {
        return edgeService.setWorkingDate(date);
    }

    /**
     * Evaluates a global model with the provided file and parameters.
     *
     * @param file  The global model file.
     * @param date  The evaluation date(s).
     * @param lclid The logical cluster ID.
     * @return A response containing the evaluation results.
     * @throws IOException          If an error occurs during file handling.
     * @throws InterruptedException If the thread is interrupted during evaluation.
     */
    @PostMapping(EdgeEndpoints.EVALUATE_MODEL)
    public ResponseEntity<?> evaluateModel(@RequestParam("file") MultipartFile file,
                                           @RequestParam("date") List<String> date,
                                           @RequestParam("lclid") String lclid) throws IOException, InterruptedException {
        return edgeService.evaluateGlobalModel(file, date, lclid);
    }
}
