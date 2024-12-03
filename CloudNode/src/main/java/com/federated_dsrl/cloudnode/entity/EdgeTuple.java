package com.federated_dsrl.cloudnode.entity;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class EdgeTuple {
    private String name;
    private String host;
    private String mac;
}