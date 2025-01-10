package com.federated_dsrl.cloudnode.tools.traffic;

import com.federated_dsrl.cloudnode.config.DeviceManager;
import com.federated_dsrl.cloudnode.entity.CloudTraffic;
import com.federated_dsrl.cloudnode.entity.EdgeTuple;
import jakarta.servlet.*;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpServletResponseWrapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A filter for monitoring outgoing traffic to specific endpoints.
 * <p>
 * This filter intercepts responses, calculates their sizes, and tracks the outgoing traffic
 * in the {@link CloudTraffic} entity while skipping untraceable URLs defined in the filter.
 * </p>
 */
@Component
@RequiredArgsConstructor
public class CloudOutgoingTrafficFilter implements Filter {

    /**
     * The entity for managing cloud traffic data.
     */
    private final CloudTraffic cloudTraffic;

    /**
     * The manager for device configurations, including fogs and edges.
     */
    private final DeviceManager deviceManager;

    /**
     * A list of URLs that should not be tracked.
     */
    private final List<String> TRAFFIC_UNTRACEABLE_URLS = createUnTrackableList();

    /**
     * Initializes the filter. This method can be overridden if additional initialization logic is required.
     *
     * @param filterConfig configuration information for the filter.
     * @throws ServletException if an error occurs during initialization.
     */
    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
        Filter.super.init(filterConfig);
    }

    /**
     * Intercepts responses, calculates their sizes, and logs outgoing traffic for traceable URLs.
     *
     * @param servletRequest  the incoming request.
     * @param servletResponse the outgoing response.
     * @param filterChain     the filter chain to pass the request/response to the next filter or target.
     * @throws ServletException if an error occurs during request processing.
     * @throws IOException      if an I/O error occurs during request processing.
     */
    @Override
    public void doFilter(ServletRequest servletRequest, ServletResponse servletResponse, FilterChain filterChain)
            throws ServletException, IOException {

        HttpServletResponse httpResponse = (HttpServletResponse) servletResponse;
        ByteArrayPrintWriter pw = new ByteArrayPrintWriter();
        HttpServletResponse wrappedResponse = new HttpServletResponseWrapper(httpResponse) {
            @Override
            public PrintWriter getWriter() {
                return pw.getWriter();
            }

            @Override
            public ServletOutputStream getOutputStream() {
                return pw.getStream();
            }
        };

        // Pass the request and wrapped response through the filter chain
        filterChain.doFilter(servletRequest, wrappedResponse);

        // Capture and calculate response size
        byte[] responseBytes = pw.toByteArray();

        HttpServletRequest httpRequest = (HttpServletRequest) servletRequest;
        String requestURI = httpRequest.getRequestURI();

        if (isUntraceable(requestURI)) {
            double responseSizeMB = Math.abs(responseBytes.length / 1024.0); // Convert to kilobytes
            cloudTraffic.addOutgoingTraffic(responseSizeMB);
        }

        // Write the response body back to the original response
        servletResponse.getOutputStream().write(responseBytes);
    }

    /**
     * Releases any resources held by this filter.
     */
    @Override
    public void destroy() {
        Filter.super.destroy();
    }

    /**
     * Creates a list of URLs that should not be tracked for outgoing traffic.
     *
     * @return a list of untraceable URLs.
     */
    private List<String> createUnTrackableList() {
        List<String> urls = new ArrayList<>();
        if (deviceManager != null) {
            Map<String, String> fogsMap = deviceManager.getFogsMap();
            if (fogsMap != null) {
                fogsMap.forEach((fogHost, fogName) -> {
                    urls.add("http://192.168.2." + fogHost + ":8080/fog/request-elapsed-time-list");
                    urls.add("http://192.168.2." + fogHost + ":8080/fog/request-incoming-fog-traffic");
                    urls.add("http://192.168.2." + fogHost + ":8080/fog/request-outgoing-fog-traffic");

                    Map<String, List<EdgeTuple>> associatedEdgesToFogMap = deviceManager.getAssociatedEdgesToFogMap();
                    if (associatedEdgesToFogMap != null && associatedEdgesToFogMap.containsKey(fogHost)) {
                        associatedEdgesToFogMap.get(fogHost).forEach(edge -> {
                            urls.add("http://192.168.2." + edge.getHost() + ":8080/edge/request-incoming-edge-traffic");
                            urls.add("http://192.168.2." + edge.getHost() + ":8080/edge/request-outgoing-edge-traffic");
                        });
                    }
                });
            }

            Map<String, List<String>> edgesMap = deviceManager.getEdgesMap();
            if (edgesMap != null) {
                edgesMap.forEach((edgeHost, edgeInfo) ->
                        urls.add("http://192.168.2." + edgeHost + ":8080/edge/request-performance-result"));
            }
        }
        return urls;
    }

    /**
     * Determines whether a given URI should be tracked for outgoing traffic.
     *
     * @param requestURI the URI of the request.
     * @return {@code true} if the URI should not be tracked, {@code false} otherwise.
     */
    private boolean isUntraceable(String requestURI) {
        return TRAFFIC_UNTRACEABLE_URLS.stream().noneMatch(requestURI::matches);
    }

    /**
     * Helper class to capture and handle the response output.
     */
    private static class ByteArrayPrintWriter {
        private final ByteArrayOutputStream baos = new ByteArrayOutputStream();
        private final PrintWriter pw = new PrintWriter(baos);
        private final ServletOutputStream sos = new ByteArrayServletStream(baos);

        public PrintWriter getWriter() {
            return pw;
        }

        public ServletOutputStream getStream() {
            return sos;
        }

        public byte[] toByteArray() {
            pw.flush();
            return baos.toByteArray();
        }
    }

    /**
     * Custom implementation of {@link ServletOutputStream} to capture response data.
     */
    private static class ByteArrayServletStream extends ServletOutputStream {
        private final ByteArrayOutputStream baos;

        ByteArrayServletStream(ByteArrayOutputStream baos) {
            this.baos = baos;
        }

        @Override
        public void write(int b) {
            baos.write(b);
        }

        @Override
        public boolean isReady() {
            return false;
        }

        @Override
        public void setWriteListener(WriteListener writeListener) {
        }
    }
}
