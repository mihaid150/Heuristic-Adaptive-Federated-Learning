package com.federated_dsrl.fognode.tools;

import lombok.Getter;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Manages a {@link CountDownLatch} for synchronizing tasks across multiple threads or systems.
 * <p>
 * This class extends the functionality of {@link CountDownLatch} by:
 * <ul>
 *     <li>Allowing dynamic reinitialization of the latch with a new count</li>
 *     <li>Tracking which edges (tasks or systems) have already counted down</li>
 *     <li>Providing thread-safe methods for interacting with the latch</li>
 * </ul>
 * </p>
 */
@Getter
@Component
public class CountDownLatchManager {

    private CountDownLatch latch;
    private final Set<String> countedEdges;
    private final Lock lock = new ReentrantLock(); // Lock for handling thread synchronization

    /**
     * Constructs a new {@code CountDownLatchManager} with an initial latch count of 3.
     */
    public CountDownLatchManager() {
        this.latch = new CountDownLatch(3);
        this.countedEdges = new HashSet<>();
    }

    /**
     * Reinitializes the latch with a new count and clears the set of counted edges.
     *
     * @param count The new count for the latch.
     */
    public synchronized void resetLatch(int count) {
        this.latch = new CountDownLatch(count);
        this.countedEdges.clear();
    }

    /**
     * Retrieves the current count of the latch.
     *
     * @return The current count of the latch.
     */
    public Long getLatchCount() {
        return this.latch.getCount();
    }

    /**
     * Awaits until the latch count reaches zero.
     *
     * @throws InterruptedException If the current thread is interrupted while waiting.
     */
    public void await() throws InterruptedException {
        this.latch.await();
    }

    /**
     * Decrements the latch count by one, ensuring the decrement is only performed
     * once per unique edge identifier.
     *
     * @param lclid The unique identifier for the edge (e.g., a task or system).
     */
    public void countDown(String lclid) {
        lock.lock(); // Acquire the lock to ensure thread safety
        try {
            if (!countedEdges.contains(lclid)) {
                this.latch.countDown();
                countedEdges.add(lclid); // Mark this edge as having counted down
            } else {
                System.out.println("Edge " + lclid + " has already counted down, skipping.");
            }
        } finally {
            lock.unlock(); // Ensure the lock is released even if an exception occurs
        }
    }
}
