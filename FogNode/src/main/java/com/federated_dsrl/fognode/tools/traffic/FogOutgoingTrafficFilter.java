package com.federated_dsrl.fognode.tools.traffic;

import com.federated_dsrl.fognode.entity.FogTraffic;
import jakarta.servlet.*;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpServletResponseWrapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * A servlet filter that tracks outgoing traffic and updates the {@link FogTraffic} entity with the calculated size
 * of the responses.
 * <p>
 * This filter wraps the HTTP response to capture the response body, calculates its size in megabytes,
 * and updates the outgoing traffic data in {@link FogTraffic}.
 * </p>
 */
@Component
@RequiredArgsConstructor
public class FogOutgoingTrafficFilter implements Filter {
    private final FogTraffic fogTraffic;

    /**
     * Intercepts the outgoing response to calculate its size and updates the outgoing traffic in {@link FogTraffic}.
     * <p>
     * The response is wrapped using a custom {@link HttpServletResponseWrapper} to capture its output,
     * calculate the size, and then write it back to the original response.
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
        fogTraffic.addOutgoingTraffic(responseSizeMB);

        // Write the response body to the original response
        servletResponse.getOutputStream().write(responseBytes);
    }

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
     * Cleans up resources used by the filter. This implementation calls the default {@code destroy} method.
     */
    @Override
    public void destroy() {
        Filter.super.destroy();
    }

    /**
     * A helper class to capture the response output using a {@link ByteArrayOutputStream}.
     */
    private static class ByteArrayPrintWriter {
        private final ByteArrayOutputStream baos = new ByteArrayOutputStream();
        private final PrintWriter pw = new PrintWriter(baos);
        private final ServletOutputStream sos = new ByteArrayServletStream(baos);

        /**
         * Returns a {@link PrintWriter} for writing response content.
         *
         * @return The {@link PrintWriter} instance.
         */
        public PrintWriter getWriter() {
            return pw;
        }

        /**
         * Returns a {@link ServletOutputStream} for writing response content.
         *
         * @return The {@link ServletOutputStream} instance.
         */
        public ServletOutputStream getStream() {
            return sos;
        }

        /**
         * Converts the written content to a byte array.
         *
         * @return A byte array containing the response content.
         */
        public byte[] toByteArray() {
            pw.flush();
            return baos.toByteArray();
        }
    }

    /**
     * A helper class for a custom {@link ServletOutputStream} that writes content to a {@link ByteArrayOutputStream}.
     */
    private static class ByteArrayServletStream extends ServletOutputStream {
        private final ByteArrayOutputStream baos;

        /**
         * Constructs an instance of {@code ByteArrayServletStream}.
         *
         * @param baos The {@link ByteArrayOutputStream} to write to.
         */
        ByteArrayServletStream(ByteArrayOutputStream baos) {
            this.baos = baos;
        }

        @Override
        public void write(int b) {
            baos.write(b);
        }

        @Override
        public boolean isReady() {
            return true;
        }

        @Override
        public void setWriteListener(WriteListener writeListener) {
            // Not implemented
        }
    }
}
