package com.federated_dsrl.cloudnode.tools.traffic;

import com.federated_dsrl.cloudnode.entity.CloudTraffic;
import jakarta.servlet.*;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.List;

@Component
@RequiredArgsConstructor
public class CloudIncomingTrafficFilter implements Filter {
    private final CloudTraffic cloudTraffic;
    // list of urls to identify those that we want to monitor traffic
    private static final List<String> TRAFFIC_TRACKABLE_URLS = List.of("/cloud/init/*", "/cloud/receive-fog-model",
            "/cloud/daily-federation/*", "/cloud/get-cooling-temperature", "/cloud/load-system-state", "/cloud-favg/*");

    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
        Filter.super.init(filterConfig);
    }

    @Override
    public void doFilter(ServletRequest servletRequest, ServletResponse servletResponse, FilterChain filterChain) throws IOException, ServletException {
        HttpServletRequest httpRequest = (HttpServletRequest) servletRequest;
        String requestURI = httpRequest.getRequestURI();

        if (isTracked(requestURI)) {
            double requestSizeMB = Math.abs(servletRequest.getContentLength() / 1024.0);
            cloudTraffic.addIncomingTraffic(requestSizeMB);
        }
        filterChain.doFilter(servletRequest, servletResponse);
    }

    @Override
    public void destroy() {
        Filter.super.destroy();
    }
    private boolean isTracked(String requestURI) {
        return TRAFFIC_TRACKABLE_URLS.stream().anyMatch(requestURI::matches);
    }
}
