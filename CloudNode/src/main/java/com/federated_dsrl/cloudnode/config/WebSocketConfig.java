package com.federated_dsrl.cloudnode.config;

import com.federated_dsrl.cloudnode.handlers.AggregationWebSocketHandler;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

/**
 * Configuration class for enabling and managing WebSocket communication in the application.
 * <p>
 * This class defines a WebSocket endpoint for handling cloud aggregation-related communication.
 * </p>
 */
@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    private final AggregationWebSocketHandler aggregationWebSocketHandler;

    /**
     * Constructor for {@link WebSocketConfig}.
     * <p>
     * Initializes the configuration with a specific WebSocket handler.
     * </p>
     *
     * @param aggregationWebSocketHandler the handler for managing WebSocket connections and messages
     *                                    related to cloud aggregation.
     */
    public WebSocketConfig(AggregationWebSocketHandler aggregationWebSocketHandler) {
        this.aggregationWebSocketHandler = aggregationWebSocketHandler;
    }

    /**
     * Registers WebSocket handlers for the application.
     * <p>
     * Adds a WebSocket endpoint at {@code /ws/cloud-aggregation} to handle cloud aggregation messages.
     * This endpoint allows requests from any origin.
     * </p>
     *
     * @param registry the {@link WebSocketHandlerRegistry} used to register WebSocket handlers.
     */
    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(aggregationWebSocketHandler, "/ws/cloud-aggregation")
                .setAllowedOrigins("*");
        System.out.println("Registered WebSocket handler at /ws/cloud-aggregation");
    }
}
