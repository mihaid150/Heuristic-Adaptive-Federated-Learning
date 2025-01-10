package com.federated_dsrl.cloudnode.config;

import com.federated_dsrl.cloudnode.entity.CloudTraffic;
import com.federated_dsrl.cloudnode.tools.CloudCoolingSchedule;
import org.apache.commons.lang3.time.StopWatch;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Configuration class for defining and managing beans in the application context.
 * This class provides centralized management of application-wide dependencies
 * such as utilities for monitoring, traffic management, and cooling schedules.
 */
@Configuration
public class AppConfig {

    /**
     * Provides a singleton bean for the {@link StopWatch} used for measuring elapsed time in various processes.
     *
     * @return A new instance of {@link StopWatch}.
     */
    @Bean
    public StopWatch stopWatch() {return new StopWatch();}

    /**
     * Provides a singleton bean for managing cloud traffic.
     * This bean is used to monitor and manage incoming and outgoing traffic in the cloud node.
     *
     * @return A new instance of {@link CloudTraffic}.
     */
    @Bean
    public CloudTraffic cloudTraffic() {
        return new CloudTraffic();
    }

    /**
     * Provides a singleton bean for managing the cooling schedule in the cloud node.
     * The cooling schedule controls the operational temperature of the cloud infrastructure.
     *
     * @return A new instance of {@link CloudCoolingSchedule}.
     */
    @Bean
    public CloudCoolingSchedule coolingSchedule() {return new CloudCoolingSchedule();}
}
