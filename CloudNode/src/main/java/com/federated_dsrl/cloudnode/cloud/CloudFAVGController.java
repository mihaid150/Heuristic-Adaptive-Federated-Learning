package com.federated_dsrl.cloudnode.cloud;

import com.federated_dsrl.cloudnode.config.CloudEndpoints;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

@RequestMapping(CloudEndpoints.CLOUD_FAVG_MAPPING)
@RestController
@RequiredArgsConstructor
public class CloudFAVGController {
    private final CloudFAVGService cloudFAVGService;

    @PostMapping(CloudEndpoints.INIT)
    public ResponseEntity<?> initializeGlobalModel() throws IOException, InterruptedException {
        return cloudFAVGService.initializeGlobalProcess();
    }

    @PostMapping(CloudEndpoints.DAILY_FEDERATION)
    public ResponseEntity<?> dailyFederation(@PathVariable String date){
        try{
            System.out.println("Received date for daily federation: " + date + "...");
            cloudFAVGService.dailyFederation(date);
            return ResponseEntity.ok("Successfully daily federation for " + date);
        } catch (Exception e){
            return ResponseEntity.status(500).body("Failed to daily federation: " + e.getMessage());
        }
    }
}
