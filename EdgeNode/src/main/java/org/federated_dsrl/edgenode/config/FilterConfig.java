package org.federated_dsrl.edgenode.config;

import org.federated_dsrl.edgenode.entity.EdgeTrafficManager;
import org.federated_dsrl.edgenode.tools.traffic.EdgeIncomingTrafficFilter;
import org.federated_dsrl.edgenode.tools.traffic.EdgeOutgoingTrafficFilter;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Configuration class for registering traffic filters in the edge node system.
 * <p>
 * This class registers and configures filters for monitoring incoming and outgoing traffic
 * for edge nodes.
 * </p>
 */
@Configuration
public class FilterConfig {

    /**
     * Registers the {@link EdgeIncomingTrafficFilter} to monitor incoming traffic for the edge node.
     *
     * @param edgeTrafficManager the traffic manager responsible for tracking incoming traffic.
     * @return the {@link FilterRegistrationBean} for the incoming traffic filter.
     */
    @Bean
    public FilterRegistrationBean<EdgeIncomingTrafficFilter> incomingTrafficFilter(EdgeTrafficManager edgeTrafficManager) {
        FilterRegistrationBean<EdgeIncomingTrafficFilter> registrationBean = new FilterRegistrationBean<>();
        registrationBean.setFilter(new EdgeIncomingTrafficFilter(edgeTrafficManager));
        registrationBean.addUrlPatterns("/*");
        return registrationBean;
    }

    /**
     * Registers the {@link EdgeOutgoingTrafficFilter} to monitor outgoing traffic for the edge node.
     *
     * @param edgeTrafficManager the traffic manager responsible for tracking outgoing traffic.
     * @return the {@link FilterRegistrationBean} for the outgoing traffic filter.
     */
    @Bean
    public FilterRegistrationBean<EdgeOutgoingTrafficFilter> outgoingTrafficFilter(EdgeTrafficManager edgeTrafficManager) {
        FilterRegistrationBean<EdgeOutgoingTrafficFilter> registrationBean = new FilterRegistrationBean<>();
        registrationBean.setFilter(new EdgeOutgoingTrafficFilter(edgeTrafficManager));
        registrationBean.addUrlPatterns("/*");
        return registrationBean;
    }
}
