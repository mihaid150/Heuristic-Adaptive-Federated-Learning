package com.federated_dsrl.cloudnode.cloud;

import com.federated_dsrl.cloudnode.config.CloudEndpoints;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RequestMapping(CloudEndpoints.CLOUD_MAPPING)
@RestController
@RequiredArgsConstructor
public class CloudController {
    private final CloudService cloudService;

    @PostMapping(CloudEndpoints.INIT_CACHE) // TODO maybe change the request path from INIT_CACHE
    public ResponseEntity<?> initializeGlobalModel(@PathVariable Boolean isCacheActive, @PathVariable String
            geneticEvaluationStrategy, @PathVariable String modelType) throws IOException,
            InterruptedException {
        return cloudService.initializeGlobalProcess(isCacheActive, geneticEvaluationStrategy, modelType);
    }

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

    @PostMapping(CloudEndpoints.CREATE_ELAPSED_TIME_CHART)
    public ResponseEntity<?> createElapsedTimeChart() {
        try {
            return cloudService.createElapsedTimeChart();
        } catch (Exception e) {
            System.err.println("Error from creating elapsed time chart: " + e.getMessage());
            return ResponseEntity.status(500).body("Failed to create elapsed time chart: " + e.getMessage());
        }
    }

    @GetMapping(CloudEndpoints.GET_COOLING_TEMPERATURE)
    public ResponseEntity<?> getCoolingTemperature() {
        return cloudService.getCoolingTemperature();
    }

    @PostMapping(CloudEndpoints.CREATE_ELAPSED_TIME_CHART_FOG_LAYER)
    public ResponseEntity<?> createElapsedTimeChartFogLayer() {
        try {
            return cloudService.createElapsedTimeChartFogLayer();
        } catch (Exception e) {
            System.err.println("Error from creating elapsed time chart fog layer: " + e.getMessage());
            return ResponseEntity.status(500).body("Failed to create elapsed time chart fog layer: " + e.getMessage());
        }
    }

    @PostMapping(CloudEndpoints.CREATE_TRAFFIC_CHART)
    public ResponseEntity<?> createTrafficChart() {
        try {
            return cloudService.createTrafficChart();
        } catch (Exception e) {
            System.err.println("Error from creating traffic chart: " + e.getMessage());
            return ResponseEntity.status(500).body("Failed to create traffic chart: " + e.getMessage());
        }
    }

    @PostMapping(CloudEndpoints.CREATE_PERFORMANCE_CHART)
    public ResponseEntity<?> createPerformanceChart() {
        return cloudService.createPerformanceChart();
    }

    @PostMapping(CloudEndpoints.LOAD_SYSTEM_STATE)
    public ResponseEntity<?> loadSystemState() {
        return cloudService.loadSystemState();
    }
}
