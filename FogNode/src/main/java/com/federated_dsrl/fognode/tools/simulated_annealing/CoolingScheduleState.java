package com.federated_dsrl.fognode.tools.simulated_annealing;

/**
 * Represents the states of the {@link FogCoolingSchedule} during its lifecycle.
 * <p>
 * The states dictate the behavior of the cooling process:
 * <ul>
 *     <li>{@code INIT} - The initial state before the cooling process starts.</li>
 *     <li>{@code OPERATIONAL} - Indicates the cooling process is actively reducing temperature.</li>
 *     <li>{@code IDLE} - Represents an inactive state, either before the process starts or after cooling is complete.</li>
 * </ul>
 * </p>
 */
public enum CoolingScheduleState {
    /**
     * The initial state before the cooling process starts.
     */
    INIT,

    /**
     * Indicates the cooling process is actively reducing temperature.
     */
    OPERATIONAL,

    /**
     * Represents an inactive state, either before the process starts or after cooling is complete.
     */
    IDLE
}
