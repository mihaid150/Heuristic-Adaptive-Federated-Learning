package org.federated_dsrl.edgenode.utils;

import org.springframework.stereotype.Component;

import java.io.File;

/**
 * Utility class for edge service operations.
 */
@Component
public class EdgeServiceUtils {

    /**
     * Deletes all JSON files from the "/app/cache_json" directory.
     * <p>
     * Logs messages for deleted or non-existent files.
     * </p>
     */
    public void deleteCacheJsonFolderContent() {
        File directory = new File("/app/cache_json");
        if (!directory.exists() || !directory.isDirectory()) {
            System.err.println("Invalid directory, unable to delete fog model files.");
            return;
        }

        File[] cacheJsonFiles = directory.listFiles((dir, name) -> name.endsWith(".json"));
        if (cacheJsonFiles == null || cacheJsonFiles.length == 0) {
            System.err.println("No json files found to delete.");
            return;
        }

        for (File jsonFile : cacheJsonFiles) {
            if (jsonFile.delete()) {
                System.out.printf("Deleted json file: " + jsonFile.getName());
            } else {
                System.err.println("Failed to delete json file:" + jsonFile.getName());
            }
        }
    }
}
