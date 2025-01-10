package com.federated_dsrl.cloudnode;

import com.federated_dsrl.cloudnode.cloud.CloudService;
import org.apache.catalina.connector.Connector;
import org.apache.coyote.ajp.AbstractAjpProtocol;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.embedded.tomcat.TomcatServletWebServerFactory;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * The main entry point for the Cloud Node application.
 * This Spring Boot application enables scheduling and sets up additional configurations
 * like an AJP connector and command-line runners.
 */
@SpringBootApplication
@EnableScheduling
public class CloudNodeApplication {

    /**
     * Main method to start the Spring Boot application.
     *
     * @param args Command-line arguments.
     */
    public static void main(String[] args) {
        SpringApplication.run(CloudNodeApplication.class, args);
    }

    /**
     * Configures the Tomcat servlet container with an additional AJP connector.
     *
     * @return A configured {@link TomcatServletWebServerFactory} instance.
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

    /**
     * Defines a {@link CommandLineRunner} bean to execute tasks during the application startup.
     * It initializes and starts monitoring for received fog models.
     *
     * @param cxt The application context used to access Spring-managed beans.
     * @return A {@link CommandLineRunner} implementation.
     */
    @Bean
    public CommandLineRunner commandLineRunner(ApplicationContext cxt) {
        return args -> {
            System.out.println("CommandLineRunner is running.");
            CloudService cloudService = cxt.getBean(CloudService.class);
            cloudService.startMonitoringReceivedFogModels();
        };
    }
}
