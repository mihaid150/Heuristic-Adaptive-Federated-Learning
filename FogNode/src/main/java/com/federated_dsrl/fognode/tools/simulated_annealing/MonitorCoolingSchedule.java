package com.federated_dsrl.fognode.tools.simulated_annealing;

import com.federated_dsrl.fognode.config.DeviceManager;
import com.federated_dsrl.fognode.config.AggregationType;
import com.federated_dsrl.fognode.tools.ConcurrencyManager;
import com.federated_dsrl.fognode.utils.FogServiceUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

/**
 * Monitors the cooling schedule in the fog node environment and manages transitions between cooling states.
 * <p>
 * This component performs the following tasks:
 * <ul>
 *     <li>Prints the current temperature at regular intervals based on a configured limit.</li>
 *     <li>Checks if conditions are met for sending the aggregated fog model to the cloud.</li>
 *     <li>Resets the cooling schedule counters and triggers cloud model transmission.</li>
 * </ul>
 * </p>
 * <p>
 * The class is designed to be executed as a separate thread or task by implementing {@link Runnable}.
 * </p>
 */
@Component
@RequiredArgsConstructor
public class MonitorCoolingSchedule implements Runnable {
    private final ConcurrencyManager concurrencyManager;
    private final FogCoolingSchedule fogCoolingSchedule;
    private final DeviceManager deviceManager;
    private final FogServiceUtils fogServiceUtils;

    /**
     * Executes the monitoring logic for the cooling schedule.
     * <p>
     * The method performs the following operations:
     * <ul>
     *     <li>Logs the current temperature if the printing limit has not been reached.</li>
     *     <li>Checks whether all conditions for model transmission to the cloud are satisfied:
     *         <ul>
     *             <li>At least one aggregation has been performed.</li>
     *             <li>There are available edge devices.</li>
     *             <li>All edge devices except one have reached their operational states.</li>
     *             <li>The cooling threshold has been reached.</li>
     *         </ul>
     *     </li>
     *     <li>Sends the fog model to the cloud with the specified aggregation type if conditions are met.</li>
     *     <li>Resets relevant counters and triggers statistics collection for cloud operations.</li>
     * </ul>
     * </p>
     */
    @Override
    public void run() {
        try {
            // Log current temperature
            if (concurrencyManager.getPrinterCounter().get() < concurrencyManager.getAllowedPrintingTimes()) {
                System.out.println("Monitoring cooling schedule. Current temperature: " +
                        fogCoolingSchedule.getTemperature());
                concurrencyManager.getPrinterCounter().getAndIncrement();
            }

            // Check conditions for model transmission to the cloud
            if (Boolean.TRUE.equals(concurrencyManager.getAtLeastOneAggregation()) && !deviceManager.getEdges()
                    .isEmpty() && (fogCoolingSchedule.getOperationalStatesCounter().equals(deviceManager.getEdges()
                    .size() - 1) ||
                    fogCoolingSchedule.getTemperature() < fogCoolingSchedule.getCOOLING_THRESHOLD())) {
                System.out.println("Conditions met for sending fog model to cloud.");
                fogServiceUtils.sendFogModelToCloud(AggregationType.GENETIC);
                fogCoolingSchedule.resetCounter();
                concurrencyManager.getPrinterCounter().set(0);

                fogServiceUtils.statisticsHelper();
            }
        } catch (Exception e) {
            System.out.println("Exception in MonitorCoolingSchedule: " + e.getMessage());
        }
    }
}
