package org.federated_dsrl.edgenode.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * Represents energy data collected from a device, including the device identifier,
 * data standard, timestamp, and energy value.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class EnergyData {

    /**
     * The MAC address of the device that generated the energy data.
     */
    private String mac;

    /**
     * The standard or protocol used for collecting the energy data.
     */
    private String standard;

    /**
     * The date and time when the energy data was recorded.
     */
    private LocalDateTime datetime;

    /**
     * The energy value recorded by the device.
     */
    private Double value;
}
