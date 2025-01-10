package com.federated_dsrl.fognode.entity;

import lombok.Data;
import org.springframework.stereotype.Component;

/**
 * Represents the elapsed time for specific operations during a single fog iteration
 * in the federated learning framework.
 * <p>
 * This class is used to store the duration of the genetic evaluation process
 * within a fog iteration.
 * </p>
 */
@Data
@Component
public class ElapsedTimeOneFogIteration {

    /**
     * The time taken (in seconds) for the genetic evaluation process during one fog iteration.
     */
    private Double timeGeneticEvaluation;
}
