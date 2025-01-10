package com.federated_dsrl.cloudnode.cloud;

import com.federated_dsrl.cloudnode.config.CloudEndpoints;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

/**
 * Controller for handling cloud operations in the federated learning system.
 */
@RequestMapping(CloudEndpoints.CLOUD_MAPPING)
@RestController
@RequiredArgsConstructor
public class CloudController {
    private final CloudService cloudService;

    /**
     * Initializes the global model and process.
     *
     * @param isCacheActive             Whether the cache is active.
     * @param geneticEvaluationStrategy Strategy used for genetic evaluation.
     * @param modelType                 Type of the model being initialized.
     * @return ResponseEntity indicating success or failure.
     * @throws IOException          In case of input/output errors.
     * @throws InterruptedException In case the process is interrupted.
     */
    @PostMapping(CloudEndpoints.INIT_CACHE) // TODO maybe change the request path from INIT_CACHE
    public ResponseEntity<?> initializeGlobalModel(@PathVariable Boolean isCacheActive, @PathVariable String
            geneticEvaluationStrategy, @PathVariable String modelType) throws IOException,
            InterruptedException {
        return cloudService.initializeGlobalProcess(isCacheActive, geneticEvaluationStrategy, modelType);
    }

    /**
     * Receives results and an aggregated model file from a fog node.
     *
     * @param result      Result of the fog processing (optional).
     * @param currentDate Current date in YYYY-MM-DD format.
     * @param name        Name of the fog node.
     * @param file        Model file.
     * @return ResponseEntity indicating success or failure.
     */
    @PostMapping(CloudEndpoints.RECEIVE_FOG_MODEL)
    public ResponseEntity<?> receiveResultFromFog(@RequestParam(value = "result", required = false) String result,
                                                  @RequestParam("current_date") String currentDate,
                                                  @RequestParam("fog_name") String name,
                                                  @RequestPart("file") MultipartFile file) {
        try {
            cloudService.processReceivedResultFromFog(result, currentDate, name, file);
            return ResponseEntity.ok("Result and model received successfully");
        } catch (IOException e) {
            return ResponseEntity.status(500).body("Failed to receive result and model: " + e.getMessage());
        }
    }

    /**
     * Executes the daily federation process.
     *
     * @param date                     Federation date in YYYY-MM-DD format.
     * @param isCacheActive            Whether the cache is active.
     * @param geneticEvaluationStrategy Strategy used for genetic evaluation.
     * @return ResponseEntity indicating success or failure.
     */
    @PostMapping(CloudEndpoints.DAILY_FEDERATION_CACHE)
    public ResponseEntity<?> executeDailyFederation(@PathVariable String date, @PathVariable Boolean isCacheActive,
                                                    @PathVariable String geneticEvaluationStrategy) {
        try {
            cloudService.dailyFederation(date, isCacheActive, geneticEvaluationStrategy);
            return ResponseEntity.ok("Successfully daily federation for " + date);
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Failed to daily federation: " + e.getMessage());
        }
    }

    /**
     * Creates an elapsed time chart.
     *
     * @return ResponseEntity indicating success or failure.
     */
    @PostMapping(CloudEndpoints.CREATE_ELAPSED_TIME_CHART)
    public ResponseEntity<?> createElapsedTimeChart() {
        try {
            return cloudService.createElapsedTimeChart();
        } catch (Exception e) {
            System.err.println("Error from creating elapsed time chart: " + e.getMessage());
            return ResponseEntity.status(500).body("Failed to create elapsed time chart: " + e.getMessage());
        }
    }

    /**
     * Retrieves the cooling temperature.
     *
     * @return ResponseEntity with the cooling temperature.
     */
    @GetMapping(CloudEndpoints.GET_COOLING_TEMPERATURE)
    public ResponseEntity<?> getCoolingTemperature() {
        return cloudService.getCoolingTemperature();
    }

    /**
     * Creates an elapsed time chart for the fog layer.
     *
     * @return ResponseEntity indicating success or failure.
     */
    @PostMapping(CloudEndpoints.CREATE_ELAPSED_TIME_CHART_FOG_LAYER)
    public ResponseEntity<?> createElapsedTimeChartFogLayer() {
        try {
            return cloudService.createElapsedTimeChartFogLayer();
        } catch (Exception e) {
            System.err.println("Error from creating elapsed time chart fog layer: " + e.getMessage());
            return ResponseEntity.status(500).body("Failed to create elapsed time chart fog layer: " + e.getMessage());
        }
    }

    /**
     * Creates a traffic chart.
     *
     * @return ResponseEntity indicating success or failure.
     */
    @PostMapping(CloudEndpoints.CREATE_TRAFFIC_CHART)
    public ResponseEntity<?> createTrafficChart() {
        try {
            return cloudService.createTrafficChart();
        } catch (Exception e) {
            System.err.println("Error from creating traffic chart: " + e.getMessage());
            return ResponseEntity.status(500).body("Failed to create traffic chart: " + e.getMessage());
        }
    }

    /**
     * Creates a performance chart.
     *
     * @return ResponseEntity indicating success or failure.
     */
    @PostMapping(CloudEndpoints.CREATE_PERFORMANCE_CHART)
    public ResponseEntity<?> createPerformanceChart() {
        return cloudService.createPerformanceChart();
    }

    /**
     * Loads the system state.
     *
     * @return ResponseEntity indicating success or failure.
     */
    @PostMapping(CloudEndpoints.LOAD_SYSTEM_STATE)
    public ResponseEntity<?> loadSystemState() {
        return cloudService.loadSystemState();
    }
}
