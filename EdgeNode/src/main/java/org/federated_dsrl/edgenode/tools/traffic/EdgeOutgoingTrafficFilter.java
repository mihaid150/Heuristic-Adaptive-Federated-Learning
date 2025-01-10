package org.federated_dsrl.edgenode.tools.traffic;

import jakarta.servlet.*;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpServletResponseWrapper;
import lombok.RequiredArgsConstructor;
import org.federated_dsrl.edgenode.entity.EdgeTrafficManager;
import org.springframework.stereotype.Component;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * A filter for tracking outgoing traffic from the edge node. It calculates the size of outgoing
 * responses and updates the {@link EdgeTrafficManager} with the data.
 */
@Component
@RequiredArgsConstructor
public class EdgeOutgoingTrafficFilter implements Filter {

    /**
     * Manages traffic data for incoming and outgoing requests.
     */
    private final EdgeTrafficManager edgeTrafficManager;

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
     * Processes outgoing responses and calculates the size of traffic.
     * Captures the response body, calculates its size in MB, and logs it to the traffic manager.
     *
     * @param servletRequest  the incoming request.
     * @param servletResponse the outgoing response.
     * @param filterChain     the filter chain.
     * @throws IOException      if an I/O error occurs during processing.
     * @throws ServletException if a servlet error occurs during processing.
     */
    @Override
    public void doFilter(ServletRequest servletRequest, ServletResponse servletResponse, FilterChain filterChain)
            throws ServletException, IOException {
        // Create a custom response wrapper
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

        // Pass the request and the wrapped response to the next filter in the chain
        filterChain.doFilter(servletRequest, wrappedResponse);

        // Calculate the size of the response body
        byte[] responseBytes = pw.toByteArray();
        double responseSizeMB = Math.abs(responseBytes.length / 1024.0);

        // Add to the outgoing traffic
        edgeTrafficManager.addOutgoingTrafficRemastered(responseSizeMB);

        // Write the response body to the original response
        servletResponse.getOutputStream().write(responseBytes);
    }

    /**
     * Destroys the filter and performs cleanup if needed.
     */
    @Override
    public void destroy() {
        Filter.super.destroy();
    }

    /**
     * Helper class to capture the response output. It acts as a wrapper around the response's
     * output stream and writer, allowing the response data to be intercepted and measured.
     */
    private static class ByteArrayPrintWriter {
        private final ByteArrayOutputStream baos = new ByteArrayOutputStream();
        private final PrintWriter pw = new PrintWriter(baos);
        private final ServletOutputStream sos = new ByteArrayServletStream(baos);

        /**
         * Returns a {@link PrintWriter} to capture the response's text output.
         *
         * @return a PrintWriter for capturing response text.
         */
        public PrintWriter getWriter() {
            return pw;
        }

        /**
         * Returns a {@link ServletOutputStream} to capture the response's binary output.
         *
         * @return a ServletOutputStream for capturing response data.
         */
        public ServletOutputStream getStream() {
            return sos;
        }

        /**
         * Converts the captured response output to a byte array.
         *
         * @return the captured response as a byte array.
         */
        public byte[] toByteArray() {
            pw.flush();
            return baos.toByteArray();
        }
    }

    /**
     * A custom implementation of {@link ServletOutputStream} that captures data written to the stream.
     */
    private static class ByteArrayServletStream extends ServletOutputStream {
        private final ByteArrayOutputStream baos;

        /**
         * Constructs a new {@link ByteArrayServletStream} that writes to the provided byte array stream.
         *
         * @param baos the byte array stream to capture the output.
         */
        ByteArrayServletStream(ByteArrayOutputStream baos) {
            this.baos = baos;
        }

        /**
         * Writes a single byte to the stream.
         *
         * @param b the byte to write.
         */
        @Override
        public void write(int b) {
            baos.write(b);
        }

        /**
         * Indicates whether the stream is ready for writing.
         *
         * @return {@code false} as this implementation does not support asynchronous writing.
         */
        @Override
        public boolean isReady() {
            return false;
        }

        /**
         * Sets a {@link WriteListener} for non-blocking writes. This implementation does nothing.
         *
         * @param writeListener the listener for write events.
         */
        @Override
        public void setWriteListener(WriteListener writeListener) {
            // No-op
        }
    }
}
