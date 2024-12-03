package com.federated_dsrl.cloudnode.config;

import com.federated_dsrl.cloudnode.handlers.AggregationWebSocketHandler;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    private final AggregationWebSocketHandler aggregationWebSocketHandler;

    public WebSocketConfig(AggregationWebSocketHandler aggregationWebSocketHandler) {
        this.aggregationWebSocketHandler = aggregationWebSocketHandler;
    }

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(aggregationWebSocketHandler, "/ws/cloud-aggregation")
                .setAllowedOrigins("*");
        System.out.println("Registered websocket handler at /ws/cloud-aggregation");
    }
}

