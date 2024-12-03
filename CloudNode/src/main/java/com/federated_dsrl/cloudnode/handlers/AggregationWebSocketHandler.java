package com.federated_dsrl.cloudnode.handlers;

import org.springframework.lang.NonNull;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

@Component
public class AggregationWebSocketHandler extends TextWebSocketHandler {
    private final List<WebSocketSession> sessions = new CopyOnWriteArrayList<>();

    @Override
    public void afterConnectionEstablished(@NonNull WebSocketSession session) {
        sessions.add(session);
        System.out.println("WebSocket connection established, session id: " + session.getId());
    }


    @Override
    public void afterConnectionClosed(@NonNull WebSocketSession session, @NonNull CloseStatus status) {
        sessions.remove(session);
        System.out.println("WebSocket connection closed, session id: " + session.getId() + " status: " + status);
    }

    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) throws Exception {
        System.out.println("Transport error in WebSocket session ID: " + session.getId() + " " + exception.getMessage());
        super.handleTransportError(session, exception);
    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) {
        System.out.println("Received message from Session ID: {}. Message: " + session.getId() + " " +
                message.getPayload());
    }

    public void notifyAggregationComplete(String message) {
        System.out.println("Notifying all sessions about aggregation completion with message: " + message);
        for (WebSocketSession session : sessions) {
            if (session.isOpen()) {
                try {
                    session.sendMessage(new TextMessage(message));
                    System.out.println("Sent message to session id: " + session.getId());
                } catch (IOException e) {
                    System.out.println("Error sending message to session id: " + session.getId());
                    throw new RuntimeException(e);
                }
            } else {
                System.out.println("Attempted to send message to closed session: " + session.getId());
            }
        }
    }
}
