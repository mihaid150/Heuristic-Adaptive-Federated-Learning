package com.federated_dsrl.fognode;

import com.federated_dsrl.fognode.fog.FogService;
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
 * The entry point for the Fog Node application.
 * <p>
 * This class initializes the application, configures AJP connectors for Tomcat,
 * and starts scheduled tasks for the fog service.
 * </p>
 */
@SpringBootApplication
@EnableScheduling
public class FogNodeApplication {

	/**
	 * The main method that starts the Spring Boot application.
	 *
	 * @param args command-line arguments
	 */
	public static void main(String[] args) {
		SpringApplication.run(FogNodeApplication.class, args);
	}

	/**
	 * Configures the Tomcat servlet container with an AJP connector.
	 * <p>
	 * This method adds support for the AJP protocol, allowing the server to handle requests
	 * from a reverse proxy like Apache HTTP Server.
	 * </p>
	 *
	 * @return a configured {@link TomcatServletWebServerFactory}
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
	 * A {@link CommandLineRunner} bean that runs after the application context is initialized.
	 * <p>
	 * This method retrieves the {@link FogService} bean and starts monitoring the cooling schedule.
	 * </p>
	 *
	 * @param ctx the Spring application context
	 * @return a {@link CommandLineRunner} that starts the fog service
	 */
	@Bean
	public CommandLineRunner commandLineRunner(ApplicationContext ctx) {
		return args -> {
			System.out.println("CommandLineRunner is running.");
			FogService fogService = ctx.getBean(FogService.class);
			fogService.startMonitoringCoolingSchedule();
		};
	}
}
