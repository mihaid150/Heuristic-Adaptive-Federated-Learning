package com.federated_dsrl.cloudnode.entity;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class EdgePerformanceResult {
    private Double edgeModelPerformance;
    private Double fogModelPerformance;
    private String date;
}
