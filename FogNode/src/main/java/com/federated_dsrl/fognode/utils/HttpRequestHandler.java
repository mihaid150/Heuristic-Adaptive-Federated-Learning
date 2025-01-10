package com.federated_dsrl.fognode.utils;

import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

/**
 * A utility class for handling HTTP POST requests using {@link RestTemplate}.
 * <p>
 * Provides methods to send POST requests with a request body and headers,
 * optionally returning a response of a specified type.
 * </p>
 */
@Component
@RequiredArgsConstructor
public class HttpRequestHandler {

    private final RestTemplate restTemplate = new RestTemplate();

    /**
     * Sends a POST request to the specified URL with the provided body and headers.
     *
     * @param url     The URL to send the POST request to.
     * @param body    The body of the POST request as a {@link MultiValueMap}.
     * @param headers The HTTP headers for the request.
     */
    public void sendPostRequest(String url, MultiValueMap<String, ?> body, HttpHeaders headers) {
        HttpEntity<MultiValueMap<String, ?>> requestEntity = new HttpEntity<>(body, headers);
        restTemplate.postForEntity(url, requestEntity, String.class);
    }

    /**
     * Sends a POST request to the specified URL with the provided body and headers,
     * returning a response of the specified type.
     *
     * @param <T>         The type of the response body.
     * @param url         The URL to send the POST request to.
     * @param body        The body of the POST request as a {@link MultiValueMap}.
     * @param headers     The HTTP headers for the request.
     * @param responseType The class of the response type.
     * @return The response body as an instance of {@code responseType}.
     */
    public <T> T sendPostRequest(String url, MultiValueMap<String, ?> body, HttpHeaders headers, Class<T> responseType) {
        HttpEntity<MultiValueMap<String, ?>> requestEntity = new HttpEntity<>(body, headers);
        ResponseEntity<T> response = restTemplate.postForEntity(url, requestEntity, responseType);
        return response.getBody();
    }
}
