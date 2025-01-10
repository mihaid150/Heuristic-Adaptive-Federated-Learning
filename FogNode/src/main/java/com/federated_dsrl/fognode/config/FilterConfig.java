package com.federated_dsrl.fognode.config;

import com.federated_dsrl.fognode.tools.traffic.FogIncomingTrafficFilter;
import com.federated_dsrl.fognode.entity.FogTraffic;
import com.federated_dsrl.fognode.tools.traffic.FogOutgoingTrafficFilter;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Configuration class for registering servlet filters used to manage incoming and outgoing traffic
 * for the fog node in the federated learning framework.
 * <p>
 * Filters are used to intercept HTTP requests and responses, enabling the implementation of traffic
 * control logic, monitoring, and custom processing.
 * </p>
 */
@Configuration
public class FilterConfig {

    /**
     * Registers a filter for managing incoming traffic to the fog node.
     * <p>
     * The {@link FogIncomingTrafficFilter} is applied to all incoming HTTP requests,
     * allowing monitoring and processing of traffic before it reaches the application.
     * </p>
     *
     * @param fogTraffic the {@link FogTraffic} instance used to monitor traffic
     * @return a {@link FilterRegistrationBean} for the incoming traffic filter
     */
    @Bean
    public FilterRegistrationBean<FogIncomingTrafficFilter> incomingTrafficFilter(FogTraffic fogTraffic) {
        FilterRegistrationBean<FogIncomingTrafficFilter> registrationBean = new FilterRegistrationBean<>();
        registrationBean.setFilter(new FogIncomingTrafficFilter(fogTraffic));
        registrationBean.addUrlPatterns("/*"); // Apply filter to all URL patterns
        return registrationBean;
    }

    /**
     * Registers a filter for managing outgoing traffic from the fog node.
     * <p>
     * The {@link FogOutgoingTrafficFilter} is applied to all outgoing HTTP responses,
     * enabling monitoring and custom processing of traffic before it leaves the application.
     * </p>
     *
     * @param fogTraffic the {@link FogTraffic} instance used to monitor traffic
     * @return a {@link FilterRegistrationBean} for the outgoing traffic filter
     */
    @Bean
    public FilterRegistrationBean<FogOutgoingTrafficFilter> outgoingTrafficFilter(FogTraffic fogTraffic) {
        FilterRegistrationBean<FogOutgoingTrafficFilter> registrationBean = new FilterRegistrationBean<>();
        registrationBean.setFilter(new FogOutgoingTrafficFilter(fogTraffic));
        registrationBean.addUrlPatterns("/*"); // Apply filter to all URL patterns
        return registrationBean;
    }
}
