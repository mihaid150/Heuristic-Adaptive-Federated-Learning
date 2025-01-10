package com.federated_dsrl.cloudnode.tools;

import lombok.Getter;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * Manages a simulated cooling process for the cloud node's system.
 * <p>
 * The cooling process is represented by a temperature value that decreases over time until it reaches a defined threshold.
 * The class includes functionality to start and stop the cooling process and runs on a scheduled basis.
 * </p>
 */
@Getter
@Component
public class CloudCoolingSchedule {

    /**
     * The initial temperature at the start of the cooling process.
     */
    private final Double INITIAL_TEMPERATURE = 100.0;

    /**
     * The current temperature of the system, which decreases during cooling.
     */
    private volatile Double temperature = INITIAL_TEMPERATURE;

    /**
     * Flag indicating whether the cooling process is currently operational.
     */
    private volatile Boolean isCoolingOperational = Boolean.FALSE;

    /**
     * Periodically reduces the system's temperature according to a cooling coefficient if the cooling process is active.
     * <p>
     * The cooling process stops when the temperature drops below a defined threshold.
     * </p>
     * <p>
     * This method is scheduled to run every 2 seconds.
     * </p>
     */
    @Scheduled(fixedRate = 2000)
    private void cloudCoolingScheduleThread() {
        if (!isCoolingOperational) {
            return;
        }

        synchronized (this) {
            Double COOLING_THRESHOLD = 0.01;
            if (temperature > COOLING_THRESHOLD) {
                double COOLING_COEFFICIENT = 0.987;
                temperature *= COOLING_COEFFICIENT;
            } else {
                isCoolingOperational = Boolean.FALSE;
            }
        }
    }

    /**
     * Starts the cooling process by setting the temperature to the initial value and activating cooling.
     */
    public synchronized void startCloudCoolingScheduleThread() {
        isCoolingOperational = Boolean.TRUE;
        temperature = INITIAL_TEMPERATURE;
    }

    /**
     * Stops the cooling process, preventing further temperature reduction.
     */
    public synchronized void stopCloudCoolingScheduleThread() {
        isCoolingOperational = Boolean.FALSE;
    }
}
