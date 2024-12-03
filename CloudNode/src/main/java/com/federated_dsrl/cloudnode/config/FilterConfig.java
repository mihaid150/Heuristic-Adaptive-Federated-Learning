package com.federated_dsrl.cloudnode.config;

import com.federated_dsrl.cloudnode.tools.traffic.CloudIncomingTrafficFilter;
import com.federated_dsrl.cloudnode.entity.CloudTraffic;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class FilterConfig {
    // using filters to measure the request body size

    @Bean
    public FilterRegistrationBean<CloudIncomingTrafficFilter> incomingTrafficFilter(CloudTraffic cloudTraffic) {
        FilterRegistrationBean<CloudIncomingTrafficFilter> registrationBean = new FilterRegistrationBean<>();
        registrationBean.setFilter(new CloudIncomingTrafficFilter(cloudTraffic));
        registrationBean.addUrlPatterns("/*"); // apply filter to all outgoing requests

        return registrationBean;
    }

    @Bean
    public FilterRegistrationBean<CloudIncomingTrafficFilter> outgoingTrafficFilter(CloudTraffic cloudTraffic) {
        FilterRegistrationBean<CloudIncomingTrafficFilter> registrationBean = new FilterRegistrationBean<>();
        registrationBean.setFilter(new CloudIncomingTrafficFilter(cloudTraffic));
        registrationBean.addUrlPatterns("/*"); // apply filter to all outgoing requests
        return registrationBean;
    }
}
