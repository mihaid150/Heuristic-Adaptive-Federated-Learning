package com.federated_dsrl.cloudnode.entity;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ResultEntry {
    private String lclid;
    private double originalModelPerformance;
    private double retrainedModelPerformance;
}
