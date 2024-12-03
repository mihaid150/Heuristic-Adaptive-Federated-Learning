package com.federated_dsrl.cloudnode.tools;

import lombok.Getter;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Getter
@Component
public class CoolingSchedule {
    private final Double INITIAL_TEMPERATURE = 100.0;
    private volatile Double temperature = INITIAL_TEMPERATURE;
    private volatile Boolean isCoolingOperational = Boolean.FALSE;

    @Scheduled(fixedRate = 2000)
    private void coolingSchedule() {
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

    public synchronized void startCooling() {
        isCoolingOperational = Boolean.TRUE;
        temperature = INITIAL_TEMPERATURE;
    }

    public synchronized void stopCooling() {
        isCoolingOperational = Boolean.FALSE;
    }
}
