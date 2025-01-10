package org.federated_dsrl.edgenode;

import org.apache.catalina.connector.Connector;
import org.apache.coyote.ajp.AbstractAjpProtocol;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.embedded.tomcat.TomcatServletWebServerFactory;
import org.springframework.context.annotation.Bean;

/**
 * Entry point for the Edge Node application.
 * <p>
 * Configures the application to use Spring Boot and sets up an AJP connector
 * for integration with external web servers.
 * </p>
 */
@SpringBootApplication
public class EdgeNodeApplication {

    /**
     * Main method for launching the Spring Boot application.
     *
     * @param args command-line arguments passed to the application.
     */
    public static void main(String[] args) {
        SpringApplication.run(EdgeNodeApplication.class, args);
    }

    /**
     * Configures an embedded Tomcat server with an AJP connector.
     *
     * @return a {@link TomcatServletWebServerFactory} configured with an AJP connector.
     */
    @Bean
    public TomcatServletWebServerFactory servletContainer() {
        TomcatServletWebServerFactory tomcat = new TomcatServletWebServerFactory();
        Connector ajpConnector = new Connector("AJP/1.3");
        ajpConnector.setPort(9090);
        ajpConnector.setSecure(false);
        ajpConnector.setAllowTrace(false);
        ajpConnector.setScheme("http");
        ((AbstractAjpProtocol<?>) ajpConnector.getProtocolHandler()).setSecretRequired(false);
        tomcat.addAdditionalTomcatConnectors(ajpConnector);
        return tomcat;
    }
}
