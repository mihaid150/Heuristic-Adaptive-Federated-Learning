package com.federated_dsrl.cloudnode.config;

import com.federated_dsrl.cloudnode.entity.EdgeTuple;
import lombok.Getter;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Getter
@Component
public class DeviceManager {
    private final Map<String, String> fogsMap;
    private final Map<String, List<String>> edgesMap;
    private final Map<String, List<EdgeTuple>> associatedEdgesToFogMap;

    public DeviceManager() {
        fogsMap = new HashMap<>();
        fogsMap.put("230", "fog2");
        fogsMap.put("231", "fog3");
        fogsMap.put("232", "fog4");

        edgesMap = new HashMap<>();
        edgesMap.put("221", List.of("edge5", "MAC000434"));
        edgesMap.put("222", List.of("edge6", "MAC004505"));
        edgesMap.put("225", List.of("edge7", "MAC001441"));
        edgesMap.put("223", List.of("edge8", "MAC002451"));
        edgesMap.put("226", List.of("edge9", "MAC001326"));
        edgesMap.put("227", List.of("edge10", "MAC004290"));
        edgesMap.put("224", List.of("edge11", "MAC002163"));
        edgesMap.put("228", List.of("edge12", "MAC001198"));
        edgesMap.put("229", List.of("edge13", "MAC000321"));

        associatedEdgesToFogMap = new HashMap<>();
    }
}
