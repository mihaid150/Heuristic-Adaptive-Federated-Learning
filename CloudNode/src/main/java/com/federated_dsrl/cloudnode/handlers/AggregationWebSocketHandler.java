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

/**
 * WebSocket handler for managing sessions and broadcasting messages during the aggregation process.
 * <p>
 * This handler manages WebSocket connections, listens for messages, and broadcasts notifications
 * about the status of cloud aggregation tasks.
 * </p>
 */
@Component
public class AggregationWebSocketHandler extends TextWebSocketHandler {

    /**
     * List of active WebSocket sessions. Concurrently managed to handle multiple clients.
     */
    private final List<WebSocketSession> sessions = new CopyOnWriteArrayList<>();

    /**
     * Invoked when a new WebSocket connection is established.
     *
     * @param session the WebSocket session representing the newly connected client
     */
    @Override
    public void afterConnectionEstablished(@NonNull WebSocketSession session) {
        sessions.add(session);
        System.out.println("WebSocket connection established, session id: " + session.getId());
    }

    /**
     * Invoked when a WebSocket connection is closed.
     *
     * @param session the WebSocket session representing the closed connection
     * @param status the status describing the reason for the connection closure
     */
    @Override
    public void afterConnectionClosed(@NonNull WebSocketSession session, @NonNull CloseStatus status) {
        sessions.remove(session);
        System.out.println("WebSocket connection closed, session id: " + session.getId() + " status: " + status);
    }

    /**
     * Invoked when a transport error occurs in a WebSocket session.
     *
     * @param session the WebSocket session where the error occurred
     * @param exception the exception representing the transport error
     * @throws Exception if further error handling is required
     */
    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) throws Exception {
        System.out.println("Transport error in WebSocket session ID: " + session.getId() + " " + exception.getMessage());
        super.handleTransportError(session, exception);
    }

    /**
     * Handles incoming text messages from clients.
     *
     * @param session the WebSocket session from which the message was received
     * @param message the text message received
     */
    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) {
        System.out.println("Received message from Session ID: {}. Message: " + session.getId() + " " +
                message.getPayload());
    }

    /**
     * Sends a notification to all active WebSocket sessions about the completion of an aggregation process.
     *
     * @param message the message to broadcast to all connected sessions
     */
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
