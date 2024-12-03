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

@Component
@RequiredArgsConstructor
public class CloudOutgoingTrafficFilter implements Filter {
    private final CloudTraffic cloudTraffic;
    private final DeviceManager deviceManager;
    // list of urls that we do not want to track
    private final List<String> TRAFFIC_UNTRACEABLE_URLS = createUnTrackableList();

    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
        Filter.super.init(filterConfig);
    }

    @Override
    public void doFilter(ServletRequest servletRequest, ServletResponse servletResponse, FilterChain filterChain)
            throws ServletException, IOException {
        // create a custom response wrapper
        HttpServletResponse httpResponse = (HttpServletResponse) servletResponse;
        ByteArrayPrintWriter pw = new ByteArrayPrintWriter();
        HttpServletResponse wrappedResponse = new HttpServletResponseWrapper(httpResponse) {
            @Override
            public PrintWriter getWriter(){
                return pw.getWriter();
            }

            @Override
            public ServletOutputStream getOutputStream(){
                return pw.getStream();
            }
        };

        // pass the request and the wrapped response to the next filter in the chain
        filterChain.doFilter(servletRequest, wrappedResponse);

        // calculate the size of the response body
        byte[] responseBytes = pw.toByteArray();

        HttpServletRequest httpRequest = (HttpServletRequest) servletRequest;
        String requestURI = httpRequest.getRequestURI();
        if (isUntraceable(requestURI)) {
            double responseSizeMB = Math.abs(responseBytes.length / 1024.0);

            // add to the outgoing traffic
            cloudTraffic.addOutgoingTraffic(responseSizeMB);
        }

        // write the response body to the original response
        servletResponse.getOutputStream().write(responseBytes);
    }

    @Override
    public void destroy() {
        Filter.super.destroy();
    }

    // helper class to capture the response output
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

    // helper class for custom ServletOutputStream
    private static class ByteArrayServletStream extends ServletOutputStream {
        private final ByteArrayOutputStream baos;

        ByteArrayServletStream(ByteArrayOutputStream baos) {
            this.baos = baos;
        }

        @Override
        public void write(int b){
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

    private List<String> createUnTrackableList() {
        List<String> urls = new ArrayList<>();

        if (deviceManager != null) {
            // Handling the fogsMap
            Map<String, String> fogsMap = deviceManager.getFogsMap();
            if (fogsMap != null) {
                fogsMap.forEach((fogHost, fogName) -> {
                    // Adding URLs related to fogs
                    urls.add("http://192.168.2." + fogHost + ":8080/fog/request-elapsed-time-list");
                    urls.add("http://192.168.2." + fogHost + ":8080/fog/request-incoming-fog-traffic");
                    urls.add("http://192.168.2." + fogHost + ":8080/fog/request-outgoing-fog-traffic");

                    // Handling associated edges for each fog
                    Map<String, List<EdgeTuple>> associatedEdgesToFogMap = deviceManager.getAssociatedEdgesToFogMap();
                    if (associatedEdgesToFogMap != null && associatedEdgesToFogMap.containsKey(fogHost)) {
                        associatedEdgesToFogMap.get(fogHost).forEach(edge -> {
                            urls.add("http://192.168.2." + edge.getHost() + ":8080/edge/request-incoming-edge-traffic");
                            urls.add("http://192.168.2." + edge.getHost() + ":8080/edge/request-outgoing-edge-traffic");
                        });
                    }
                });
            }

            // Handling the edgesMap
            Map<String, List<String>> edgesMap = deviceManager.getEdgesMap();
            if (edgesMap != null) {
                edgesMap.forEach((edgeHost, edgeInfo) ->
                        urls.add("http://192.168.2." + edgeHost + ":8080/edge/request-performance-result"));
            }
        }

        return urls;
    }

    private boolean isUntraceable(String requestURI) {
        return TRAFFIC_UNTRACEABLE_URLS.stream().noneMatch(requestURI::matches);
    }
}
