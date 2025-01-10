package org.federated_dsrl.edgenode.config;

import org.federated_dsrl.edgenode.entity.EdgeTrafficManager;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Configuration class for setting up application-wide beans in the Edge Node module.
 */
@Configuration
public class AppConfig {

    /**
     * Creates and provides a singleton instance of {@link EdgeTrafficManager}.
     *
     * <p>The {@code EdgeTrafficManager} is responsible for managing and monitoring
     * traffic-related metrics in the edge node system.</p>
     *
     * @return a new instance of {@code EdgeTrafficManager}.
     */
    @Bean
    public EdgeTrafficManager edgeTrafficManager() {
        return new EdgeTrafficManager();
    }
}
