package com.federated_dsrl.cloudnode.tools.traffic;

import com.federated_dsrl.cloudnode.entity.CloudTraffic;
import jakarta.servlet.*;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.List;

/**
 * A filter for monitoring incoming traffic to specific cloud endpoints.
 * <p>
 * This filter calculates the size of incoming requests for specified URLs and logs the data
 * into the {@link CloudTraffic} entity.
 * </p>
 */
@Component
@RequiredArgsConstructor
public class CloudIncomingTrafficFilter implements Filter {

    /**
     * Tracks incoming traffic details.
     */
    private final CloudTraffic cloudTraffic;

    /**
     * List of URLs to monitor for traffic tracking.
     */
    private static final List<String> TRAFFIC_TRACKABLE_URLS = List.of(
            "/cloud/init/*",
            "/cloud/receive-fog-model",
            "/cloud/daily-federation/*",
            "/cloud/get-cooling-temperature",
            "/cloud/load-system-state",
            "/cloud-favg/*"
    );

    /**
     * Initializes the filter. This method can be overridden if any initialization logic is required.
     *
     * @param filterConfig configuration information for the filter.
     * @throws ServletException if an error occurs during initialization.
     */
    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
        Filter.super.init(filterConfig);
    }

    /**
     * Intercepts incoming requests, calculates their size, and updates the {@link CloudTraffic} entity
     * if the request matches one of the tracked URLs.
     *
     * @param servletRequest  the incoming servlet request.
     * @param servletResponse the outgoing servlet response.
     * @param filterChain     the filter chain to pass the request/response to the next filter or target.
     * @throws IOException      if an I/O error occurs during request processing.
     * @throws ServletException if an error occurs during request processing.
     */
    @Override
    public void doFilter(ServletRequest servletRequest, ServletResponse servletResponse, FilterChain filterChain)
            throws IOException, ServletException {

        HttpServletRequest httpRequest = (HttpServletRequest) servletRequest;
        String requestURI = httpRequest.getRequestURI();

        if (isTracked(requestURI)) {
            double requestSizeMB = Math.abs(servletRequest.getContentLength() / 1024.0); // Convert to kilobytes
            cloudTraffic.addIncomingTraffic(requestSizeMB);
        }

        // Proceed with the next filter in the chain
        filterChain.doFilter(servletRequest, servletResponse);
    }

    /**
     * Releases any resources held by this filter.
     */
    @Override
    public void destroy() {
        Filter.super.destroy();
    }

    /**
     * Checks if the given URI matches one of the trackable URLs.
     *
     * @param requestURI the URI of the incoming request.
     * @return {@code true} if the URI matches one of the tracked URLs, {@code false} otherwise.
     */
    private boolean isTracked(String requestURI) {
        return TRAFFIC_TRACKABLE_URLS.stream().anyMatch(requestURI::matches);
    }
}
