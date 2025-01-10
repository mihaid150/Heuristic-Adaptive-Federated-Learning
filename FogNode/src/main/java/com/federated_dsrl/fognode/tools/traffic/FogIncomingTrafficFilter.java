package com.federated_dsrl.fognode.tools.traffic;

import com.federated_dsrl.fognode.entity.FogTraffic;
import jakarta.servlet.*;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.List;

/**
 * A servlet filter that tracks incoming traffic for specific URLs and updates the {@link FogTraffic} entity
 * with the calculated size of the requests.
 * <p>
 * This filter processes incoming HTTP requests, determines if the request URI matches traceable URLs, and
 * calculates the size of the incoming traffic in megabytes. If the URL is traceable, the traffic size is added
 * to the {@link FogTraffic} entity.
 * </p>
 */
@Component
@RequiredArgsConstructor
public class FogIncomingTrafficFilter implements Filter {
    private final FogTraffic fogTraffic;
    private static final List<String> TRAFFIC_TRACEABLE_URLS = List.of(
            "/fog/add-edge", "/fog/ack-cloud", "/fog/receive-global-model", "/fog/receive-edge-model",
            "/fog/request-fog-model", "/fog/edge-ready", "/fog/edge-should-proceed/*",
            "/fog/load-system-state", "/fog-favg/*"
    );

    /**
     * Initializes the filter. This implementation calls the default {@code init} method.
     *
     * @param filterConfig The filter configuration object.
     * @throws ServletException If an error occurs during initialization.
     */
    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
        Filter.super.init(filterConfig);
    }

    /**
     * Processes each incoming HTTP request and determines if the request URI is traceable.
     * <p>
     * If the URI is traceable, the size of the incoming request is calculated in megabytes and added
     * to the {@link FogTraffic} entity.
     * </p>
     *
     * @param servletRequest  The servlet request object.
     * @param servletResponse The servlet response object.
     * @param filterChain     The filter chain to continue processing the request.
     * @throws ServletException If an error occurs during request processing.
     * @throws IOException      If an I/O error occurs during request processing.
     */
    @Override
    public void doFilter(ServletRequest servletRequest, ServletResponse servletResponse, FilterChain filterChain)
            throws ServletException, IOException {
        HttpServletRequest httpRequest = (HttpServletRequest) servletRequest;
        String requestURI = httpRequest.getRequestURI();

        if (isTraceable(requestURI)) {
            double requestSizeMB = Math.abs(servletRequest.getContentLength() / 1024.0);
            fogTraffic.addIncomingTraffic(requestSizeMB);
        }
        filterChain.doFilter(servletRequest, servletResponse);
    }

    /**
     * Cleans up resources used by the filter. This implementation calls the default {@code destroy} method.
     */
    @Override
    public void destroy() {
        Filter.super.destroy();
    }

    /**
     * Checks whether a given request URI matches the predefined list of traceable URLs.
     *
     * @param requestURI The request URI to check.
     * @return {@code true} if the URI is traceable; otherwise {@code false}.
     */
    private boolean isTraceable(String requestURI) {
        return TRAFFIC_TRACEABLE_URLS.stream().anyMatch(requestURI::matches);
    }
}
