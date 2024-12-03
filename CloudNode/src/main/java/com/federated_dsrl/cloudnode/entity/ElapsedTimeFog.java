package com.federated_dsrl.cloudnode.entity;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class ElapsedTimeFog {
    private Double timeForAllEdgesReadiness;
    private Double timeGeneticEvaluation;
    private List<Double> timeReceivedEdgeModel = new ArrayList<>();
    private List<Double> timeFinishAggregation = new ArrayList<>();
}
