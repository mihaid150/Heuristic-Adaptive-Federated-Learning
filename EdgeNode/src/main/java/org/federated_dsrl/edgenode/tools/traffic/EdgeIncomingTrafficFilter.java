package org.federated_dsrl.edgenode.tools.traffic;

import jakarta.servlet.*;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.federated_dsrl.edgenode.entity.EdgeTrafficManager;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.List;

/**
 * A filter for tracking incoming traffic to the edge node. It calculates the size of incoming
 * requests for specific traceable URLs and updates the {@link EdgeTrafficManager} with the data.
 */
@Component
@RequiredArgsConstructor
public class EdgeIncomingTrafficFilter implements Filter {

    /**
     * Manages traffic data for incoming and outgoing requests.
     */
    private final EdgeTrafficManager edgeTrafficManager;

    /**
     * List of URLs for which incoming traffic will be tracked.
     */
    private static final List<String> TRAFFIC_TRACEABLE_URLS = List.of(
            "/edge/parent-fog",
            "/edge/receive-fog-model",
            "/edge/genetics-training",
            "/edge/receive-parameters",
            "/edge/set-working-date"
    );

    /**
     * Initializes the filter.
     *
     * @param filterConfig the filter configuration object.
     * @throws ServletException if an error occurs during initialization.
     */
    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
        Filter.super.init(filterConfig);
    }

    /**
     * Processes incoming requests and calculates the size of traceable traffic.
     * If the request URI matches a traceable URL, the traffic size is logged.
     *
     * @param servletRequest  the incoming request.
     * @param servletResponse the outgoing response.
     * @param filterChain     the filter chain.
     * @throws IOException      if an I/O error occurs during processing.
     * @throws ServletException if a servlet error occurs during processing.
     */
    @Override
    public void doFilter(ServletRequest servletRequest, ServletResponse servletResponse, FilterChain filterChain)
            throws IOException, ServletException {
        HttpServletRequest httpRequest = (HttpServletRequest) servletRequest;
        String requestURI = httpRequest.getRequestURI();

        if (isTraceable(requestURI)) {
            double requestSizeMB = Math.abs(servletRequest.getContentLength() / 1024.0);
            edgeTrafficManager.addIncomingTrafficRemastered(requestSizeMB);
        }
        filterChain.doFilter(servletRequest, servletResponse);
    }

    /**
     * Destroys the filter and performs cleanup if needed.
     */
    @Override
    public void destroy() {
        Filter.super.destroy();
    }

    /**
     * Checks if the given request URI matches any of the traceable URLs.
     *
     * @param requestURI the request URI to check.
     * @return {@code true} if the URI is traceable; {@code false} otherwise.
     */
    private boolean isTraceable(String requestURI) {
        return TRAFFIC_TRACEABLE_URLS.stream().anyMatch(requestURI::matches);
    }
}
