package com.federated_dsrl.cloudnode.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.lang.NonNull;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * Configuration class for enabling Cross-Origin Resource Sharing (CORS) in the application.
 * <p>
 * This configuration allows the application to handle cross-origin requests from specified origins
 * and supports a variety of HTTP methods.
 * </p>
 */
@Configuration
public class WebConfig {

    /**
     * Defines a {@link WebMvcConfigurer} bean for configuring CORS mappings.
     * <p>
     * This method customizes the CORS configuration for the application, allowing requests
     * from specified origins with a set of allowed methods, headers, and credentials.
     * </p>
     *
     * @return a {@link WebMvcConfigurer} instance with the specified CORS mappings.
     */
    @Bean
    public WebMvcConfigurer corsConfigurer() {
        return new WebMvcConfigurer() {

            /**
             * Configures CORS mappings for the application.
             * <p>
             * Allows requests from the specified origin (`<a href="http://localhost:5173">...</a>`) and permits
             * HTTP methods such as GET, POST, PUT, DELETE, and OPTIONS. All headers are allowed,
             * and credentials are supported in requests.
             * </p>
             *
             * @param registry the {@link CorsRegistry} used to configure CORS mappings.
             */
            @Override
            public void addCorsMappings(@NonNull CorsRegistry registry) {
                registry.addMapping("/**")
                        .allowedOrigins("http://localhost:5173")
                        .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
                        .allowedHeaders("*")
                        .allowCredentials(true);
            }
        };
    }
}
