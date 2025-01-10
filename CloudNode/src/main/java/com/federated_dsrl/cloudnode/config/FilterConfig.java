package com.federated_dsrl.cloudnode.config;

import com.federated_dsrl.cloudnode.tools.traffic.CloudIncomingTrafficFilter;
import com.federated_dsrl.cloudnode.entity.CloudTraffic;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Configuration class for registering servlet filters to monitor traffic
 * in the cloud node.
 * <p>Filters are used to measure request and response body sizes for
 * tracking incoming and outgoing traffic.</p>
 */
@Configuration
public class FilterConfig {

    /**
     * Registers a filter for monitoring incoming traffic to the cloud node.
     *
     * @param cloudTraffic the {@link CloudTraffic} instance to track incoming traffic metrics.
     * @return a {@link FilterRegistrationBean} configured with {@link CloudIncomingTrafficFilter}
     * for monitoring incoming traffic.
     */
    @Bean
    public FilterRegistrationBean<CloudIncomingTrafficFilter> incomingTrafficFilter(CloudTraffic cloudTraffic) {
        FilterRegistrationBean<CloudIncomingTrafficFilter> registrationBean = new FilterRegistrationBean<>();
        registrationBean.setFilter(new CloudIncomingTrafficFilter(cloudTraffic));
        registrationBean.addUrlPatterns("/*"); // Apply filter to all incoming requests
        return registrationBean;
    }

    /**
     * Registers a filter for monitoring outgoing traffic from the cloud node.
     *
     * @param cloudTraffic the {@link CloudTraffic} instance to track outgoing traffic metrics.
     * @return a {@link FilterRegistrationBean} configured with {@link CloudIncomingTrafficFilter}
     * for monitoring outgoing traffic.
     */
    @Bean
    public FilterRegistrationBean<CloudIncomingTrafficFilter> outgoingTrafficFilter(CloudTraffic cloudTraffic) {
        FilterRegistrationBean<CloudIncomingTrafficFilter> registrationBean = new FilterRegistrationBean<>();
        registrationBean.setFilter(new CloudIncomingTrafficFilter(cloudTraffic));
        registrationBean.addUrlPatterns("/*"); // Apply filter to all outgoing requests
        return registrationBean;
    }
}
