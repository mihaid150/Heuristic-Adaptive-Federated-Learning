package com.federated_dsrl.cloudnode.cloud;

import com.federated_dsrl.cloudnode.entity.CloudTraffic;
import com.federated_dsrl.cloudnode.tools.ConcurrencyManager;
import com.federated_dsrl.cloudnode.tools.MonitorReceivedFogModels;
import com.federated_dsrl.cloudnode.utils.CloudServiceUtils;
import com.federated_dsrl.cloudnode.config.AggregationType;
import com.federated_dsrl.cloudnode.config.PathManager;
import lombok.RequiredArgsConstructor;
import org.apache.commons.lang3.time.StopWatch;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.io.File;
import java.io.IOException;
import java.util.List;

@Service
@RequiredArgsConstructor
public class CloudFAVGService {
    private final CloudServiceUtils cloudServiceUtils;
    private final ConcurrencyManager concurrencyManager;
    private final CloudTraffic cloudTraffic;
    private final PathManager pathManager;
    private final MonitorReceivedFogModels monitorReceivedFogModels;
    private final HttpHeaders formHeaders = createFormHeaders();
    private final RestTemplate restTemplate = new RestTemplate();
    private final StopWatch stopWatch;

    private HttpHeaders createFormHeaders() {
        // create a global communication header for  file transmission
        // TODO check if can we a unique utility method
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        return headers;
    }

    public ResponseEntity<?> initializeGlobalProcess() throws IOException, InterruptedException {

        // reset and start the stopWatch for global elapsed time
        stopWatch.reset();
        stopWatch.start();

        // initial settings from cloud utils
        cloudServiceUtils.deleteCacheJsonFolderContent();
        cloudServiceUtils.deleteFogModelFiles();
        cloudServiceUtils.deleteResultFile();

        // clear previous saved traffic from the storage
        cloudTraffic.clearTraffic();

        // initiates the json files for monitoring the incoming/outgoing traffic
        cloudTraffic.saveTrafficToJsonFile();

        // resets all the times for measuring elapsed time
        concurrencyManager.clearAggregatedDate();

        // clear the list which holds the already dates in which we aggregated data
        concurrencyManager.clearElapsedTimeOverIterations();

        // initiates the json file for storing the aggregated dates and elapsed time
        concurrencyManager.saveCacheToJsonFile();

        // transmit the network configuration to children nodes
        cloudServiceUtils.notifyFogAboutCloud(restTemplate);
        cloudServiceUtils.associateEdgesToFogProcess(restTemplate);

        // creates the initial random (dummy) model
        // TODO: implement correctly and not hardcoded
        File dummyModel = monitorReceivedFogModels.createDummyModel("LSTM");

        // broadcast the dummy model to fogs which will forward it to children nodes
        initialBroadcastToFogs(dummyModel);
        return ResponseEntity.ok("Initialization and transmission of cloud model successfully.");
    }

    private void initialBroadcastToFogs(File dummyModel) {
        // start and end date for initial training chosen based on dataset
        List<String> dates = List.of("2012-07-09", "2013-07-09");

        // initial aggregation date
        concurrencyManager.addNewAggregatedDate("2013-07-09");

        // set to FALSE the flag for AlreadyAggregated
        concurrencyManager.setAlreadyAggregated(Boolean.FALSE);

        // call the broadcast function with arguments the dummy model and the rest of training configuration
        FileSystemResource resource = new FileSystemResource(dummyModel);
        cloudServiceUtils.broadcast(resource, dates, restTemplate, formHeaders, concurrencyManager, false,
                true, AggregationType.AVERAGE, null);
    }

    public void dailyFederation(String date) {
        // reset the stopwatch for round elapsed time measuring
        stopWatch.reset();
        stopWatch.start();

        // add to the cache the current aggregation date
        concurrencyManager.addNewAggregatedDate(date);
        // set the AlreadyAggregated flag to FALSE
        concurrencyManager.setAlreadyAggregated(Boolean.FALSE);

        // load the previous model file from previous round
        File globalModelFile = new File(pathManager.getGlobalModelPath());
        FileSystemResource resource = new FileSystemResource(globalModelFile);

        // broadcast the global model to the fogs with this round training configuration
        cloudServiceUtils.broadcast(resource, List.of(date), restTemplate, formHeaders, concurrencyManager,
                false, false, AggregationType.AVERAGE, null);
    }
}
