package com.federated_dsrl.fognode.config;

import com.federated_dsrl.fognode.entity.FogTraffic;
import com.federated_dsrl.fognode.tools.ConcurrencyManager;
import com.federated_dsrl.fognode.tools.CountDownLatchManager;
import com.federated_dsrl.fognode.tools.genetic.engine.GeneticEngine;
import com.federated_dsrl.fognode.tools.simulated_annealing.CloudCoolingSchedule;
import com.federated_dsrl.fognode.tools.simulated_annealing.FogCoolingSchedule;
import com.federated_dsrl.fognode.utils.HttpRequestHandler;
import com.federated_dsrl.fognode.utils.ModelFileHandler;
import org.apache.commons.lang3.time.StopWatch;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Application configuration class defining global singleton beans for utility classes.
 * <p>
 * This class centralizes the instantiation of components and utilities used across the application,
 * ensuring they preserve their identity and are managed by the Spring container.
 * </p>
 */
@Configuration
public class AppConfig {

    /**
     * Provides a {@link StopWatch} instance for measuring execution times in various components.
     *
     * @return a new {@link StopWatch} instance
     */
    @Bean
    public StopWatch stopWatch() {
        return new StopWatch();
    }

    /**
     * Provides a {@link FogTraffic} instance for managing traffic-related data in the fog node.
     *
     * @return a new {@link FogTraffic} instance
     */
    @Bean
    public FogTraffic fogTraffic() {
        return new FogTraffic();
    }

    /**
     * Provides a {@link FogCoolingSchedule} instance for simulated annealing optimization.
     *
     * @return a new {@link FogCoolingSchedule} instance
     */
    @Bean
    public FogCoolingSchedule coolingSchedule() {
        return new FogCoolingSchedule();
    }

    /**
     * Provides a {@link PathManager} instance for managing file paths used within the application.
     *
     * @return a new {@link PathManager} instance
     */
    @Bean
    public PathManager pathManager() {
        return new PathManager();
    }

    /**
     * Provides an {@link HttpRequestHandler} instance for handling HTTP requests.
     *
     * @return a new {@link HttpRequestHandler} instance
     */
    @Bean
    public HttpRequestHandler httpRequestHandler() {
        return new HttpRequestHandler();
    }

    /**
     * Provides a {@link ModelFileHandler} instance for managing model files, HTTP requests, and cooling schedules.
     *
     * @param pathManager       the {@link PathManager} instance
     * @param httpRequestHandler the {@link HttpRequestHandler} instance
     * @param fogCoolingSchedule   the {@link FogCoolingSchedule} instance
     * @return a new {@link ModelFileHandler} instance
     */
    @Bean
    public ModelFileHandler modelFileHandler(PathManager pathManager, HttpRequestHandler httpRequestHandler,
                                             FogCoolingSchedule fogCoolingSchedule) {
        return new ModelFileHandler(pathManager, httpRequestHandler, fogCoolingSchedule);
    }

    /**
     * Provides a {@link DeviceManager} instance for managing edge devices in the network.
     *
     * @return a new {@link DeviceManager} instance
     */
    @Bean
    public DeviceManager deviceManager() {
        return new DeviceManager();
    }

    /**
     * Provides a {@link ConcurrencyManager} instance for managing concurrency in operations.
     *
     * @return a new {@link ConcurrencyManager} instance
     */
    @Bean
    public ConcurrencyManager concurrencyManager() {
        return new ConcurrencyManager();
    }

    /**
     * Provides a {@link CloudCoolingSchedule} instance for handling cloud-level cooling schedule logic.
     *
     * @param deviceManager the {@link DeviceManager} instance
     * @return a new {@link CloudCoolingSchedule} instance
     */
    @Bean
    public CloudCoolingSchedule cloudCoolingSchedule(DeviceManager deviceManager) {
        return new CloudCoolingSchedule(deviceManager);
    }

    /**
     * Provides a {@link GeneticEngine} instance for managing genetic algorithm operations in training.
     *
     * @param modelFileHandler    the {@link ModelFileHandler} instance
     * @param deviceManager       the {@link DeviceManager} instance
     * @param concurrencyManager  the {@link ConcurrencyManager} instance
     * @param cloudCoolingSchedule the {@link CloudCoolingSchedule} instance
     * @return a new {@link GeneticEngine} instance
     */
    @Bean
    public GeneticEngine geneticEngine(ModelFileHandler modelFileHandler,
                                       DeviceManager deviceManager,
                                       ConcurrencyManager concurrencyManager,
                                       CloudCoolingSchedule cloudCoolingSchedule) {
        return new GeneticEngine(modelFileHandler, deviceManager, concurrencyManager, cloudCoolingSchedule);
    }

    /**
     * Provides a {@link CountDownLatchManager} instance for managing countdown latches in multi-threaded operations.
     *
     * @return a new {@link CountDownLatchManager} instance
     */
    @Bean
    public CountDownLatchManager countDownLatchManager() {
        return new CountDownLatchManager();
    }
}
