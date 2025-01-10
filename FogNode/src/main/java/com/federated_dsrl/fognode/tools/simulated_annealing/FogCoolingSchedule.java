package com.federated_dsrl.fognode.tools.simulated_annealing;

import lombok.Getter;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * The {@code FogCoolingSchedule} class implements a simulated cooling process
 * that decreases temperature over time until a predefined threshold is reached.
 * This class operates in different states to manage the cooling process lifecycle.
 *
 * <p>The cooling process is executed periodically using Spring's {@link Scheduled} annotation.
 * The class also includes mechanisms for resetting and incrementing operational states
 * counters, allowing for external control over the cooling lifecycle.</p>
 */
@Component
@Getter
public class FogCoolingSchedule {
    private final Double INITIAL_TEMPERATURE = 100.0;
    private final Double COOLING_THRESHOLD = 0.01;
    private volatile Double temperature = INITIAL_TEMPERATURE;
    private volatile CoolingScheduleState state = CoolingScheduleState.INIT;
    private volatile Integer operationalStatesCounter = 0;
    private final Integer allowedPrintingTimes = 5;
    private final AtomicInteger printerCounter = new AtomicInteger();

    /**
     * Periodically executes the cooling schedule.
     * <p>
     * This method is triggered at a fixed rate of 2000 milliseconds. Depending on the current state, it:
     * <ul>
     *     <li>Prints the current state (limited by {@code allowedPrintingTimes}).</li>
     *     <li>Reduces the temperature if the state is {@code OPERATIONAL} and the temperature is above the threshold.</li>
     *     <li>Transitions to the {@code IDLE} state when the cooling process is complete.</li>
     * </ul>
     * </p>
     */
    @Scheduled(fixedRate = 2000)
    private void fogCoolingScheduleThread() {
        if (printerCounter.get() < allowedPrintingTimes) {
            System.out.println("Cooling schedule triggered. Current state: " + state);
            this.printerCounter.getAndIncrement();
        }

        if (state != CoolingScheduleState.OPERATIONAL) {
            state = CoolingScheduleState.IDLE;
            return;
        }

        synchronized (this) {
            if (temperature > COOLING_THRESHOLD) {
                double COOLING_COEFFICIENT = 0.99;
                temperature *= COOLING_COEFFICIENT;
                System.out.println("Current temperature: " + temperature);
            } else { // It reached the lower bound
                state = CoolingScheduleState.IDLE;
                System.out.println("Temperature reached threshold. Cooling process transitioned to IDLE.");
            }
        }
    }

    /**
     * Starts the cooling process by resetting the temperature to its initial value and setting the state to {@code OPERATIONAL}.
     */
    public synchronized void startFogCoolingScheduleThread() {
        System.out.println("Starting cooling process...");
        temperature = INITIAL_TEMPERATURE;
        state = CoolingScheduleState.OPERATIONAL;
        this.printerCounter.set(0);
    }

    /**
     * Resets the operational states counter to zero and transitions the cooling process to the {@code IDLE} state.
     */
    public synchronized void resetCounter() {
        this.operationalStatesCounter = 0;
        this.state = CoolingScheduleState.IDLE;
        System.out.println("Operational states counter reset to 0. State set to IDLE.");
    }

    /**
     * Increments the operational states counter by one.
     */
    public synchronized void incrementCounter() {
        this.operationalStatesCounter += 1;
        System.out.println("Operational states counter incremented to " + this.operationalStatesCounter);
    }
}
