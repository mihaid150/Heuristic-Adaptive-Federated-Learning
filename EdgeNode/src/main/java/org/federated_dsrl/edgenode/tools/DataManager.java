package org.federated_dsrl.edgenode.tools;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.federated_dsrl.edgenode.config.PathManager;
import org.federated_dsrl.edgenode.entity.EnergyData;
import org.federated_dsrl.edgenode.entity.PerformanceResult;
import org.springframework.stereotype.Component;

import java.io.*;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

/**
 * Handles data management operations such as fetching and saving energy data, and managing performance results.
 */
@Component
@RequiredArgsConstructor
public class DataManager {
    /**
     * Provides access to path configurations for scripts, models, and data files.
     */
    private final PathManager pathManager;

    /**
     * Stores the list of performance results.
     */
    @Getter
    private final List<PerformanceResult> performanceResultList = new ArrayList<>();

    /**
     * Timeout duration for waiting on certain operations, in milliseconds.
     */
    @Getter
    private Long timeout = (long) (5 * 60 * 1000);

    /**
     * File path for saving performance results in JSON format.
     */
    private final String PERFORMANCE_FILE = "cache_json/performance.json";

    /**
     * Returns the current system time in milliseconds.
     *
     * @return current system time in milliseconds.
     */
    public Long getStartTime() {
        return System.currentTimeMillis();
    }

    /**
     * Checks if performance data exists for a specific date.
     *
     * @param date the date to check.
     * @return {@code true} if performance data exists for the specified date, {@code false} otherwise.
     */
    public Boolean existsPerformanceByDate(String date) {
        return performanceResultList.stream().map(PerformanceResult::getDate).toList().contains(date);
    }

    /**
     * Fetches daily energy data for a specific date and logical cluster ID (LCID).
     *
     * @param date  the date for which data is required.
     * @param lclid the logical cluster ID.
     * @return the path to the saved data JSON file.
     */
    public String provideDailyData(String date, String lclid){
        List<EnergyData> dataList = new ArrayList<>();

        File scriptFile = new File(pathManager.getSelectDataScriptPath());
        ProcessBuilder processBuilder = new ProcessBuilder(pathManager.getShortPythonExecutablePath(),
                scriptFile.getAbsolutePath(), pathManager.getCSVFilePath(lclid), date, lclid);
        processBuilder.redirectErrorStream(true);
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern(pathManager.getDateTimeFormat());

        try {
            Process process = processBuilder.start();
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            BufferedReader stdError = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String line;
            boolean isFirstLine = true;

            while ((line = bufferedReader.readLine()) != null) {
                if (isFirstLine) {
                    isFirstLine = false;
                    continue;
                }

                String[] parts = line.split(",");
                if (parts.length >= 4) {
                    LocalDateTime dateTime = LocalDateTime.parse(parts[2].trim(), formatter);
                    EnergyData data = new EnergyData(lclid, parts[1].trim(), dateTime,
                            Double.parseDouble(parts[3].trim()));
                    dataList.add(data);
                }
            }
            while ((line = stdError.readLine()) != null) {
                System.out.println("Error from select_data: " + line);
            }
            int exitCode = process.waitFor();
            System.out.println("Python script select_data exited with code:" + exitCode);
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
        return saveDataWithPython(dataList, lclid);
    }

    /**
     * Fetches energy data for a given time period and logical cluster ID (LCID).
     *
     * @param startDate the start date of the period.
     * @param endDate   the end date of the period.
     * @param lclid     the logical cluster ID.
     * @return the path to the saved data JSON file.
     */
    public String providePeriodData(String startDate, String endDate, String lclid){
        System.out.println("from provide period data method: start date: " + startDate + ", end date: " + endDate
                + ", lclid: " + lclid);
        List<EnergyData> dataList = new ArrayList<>();
        File scriptFile = new File(pathManager.getSelectMultipleDataScriptPath());
        ProcessBuilder processBuilder = new ProcessBuilder(pathManager.getShortPythonExecutablePath(),
                scriptFile.getAbsolutePath(), pathManager.getCSVFilePath(lclid), startDate, endDate);
        processBuilder.redirectErrorStream(true);
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern(pathManager.getDateTimeFormat());

        try {
            Process process = processBuilder.start();
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            BufferedReader stdError = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String line;
            boolean isFirstLine = true;

            while ((line = bufferedReader.readLine()) != null) {
                if (isFirstLine) {
                    isFirstLine = false;
                    continue;
                }
                if(line.contains("start date")) {
                    System.out.println(line);
                }

                String[] parts = line.split(",");
                if (parts.length >= 4) {
                    try {
                        LocalDateTime dateTime = LocalDateTime.parse(parts[2].trim(), formatter);
                        double value = parts[3].trim().equalsIgnoreCase("Null") ? 0.0 : Double.parseDouble(parts[3].trim());
                        EnergyData data = new EnergyData(lclid, parts[1].trim(), dateTime, value);
                        dataList.add(data);
                    } catch (Exception e) {
                        System.err.println("Error parsing line: " + line + " -> " + e.getMessage());
                    }
                }
            }

            while ((line = stdError.readLine()) != null) {
                System.out.println("Error from executing select_multiple_data: " + line);
            }

            int exitCode = process.waitFor();
            System.out.println("Python script select_multiple_data exited with code: " + exitCode);

        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
        return saveDataWithPython(dataList, lclid);
    }

    /**
     * Saves energy data to a JSON file using a Python script.
     *
     * @param dataList the list of energy data to save.
     * @param lclid    the logical cluster ID.
     * @return the path to the temporary JSON file.
     */
    private String saveDataWithPython(List<EnergyData> dataList, String lclid){
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());
        objectMapper.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
        Path tempFile;
        File scriptFile = new File(pathManager.getUtilsScriptPath());

        try {
            tempFile = Files.createTempFile(pathManager.getEnergyDataFileName(lclid), ".json");
            objectMapper.writeValue(tempFile.toFile(), dataList);

            ProcessBuilder processBuilder = new ProcessBuilder(
                    pathManager.getShortPythonExecutablePath(),
                    scriptFile.getAbsolutePath(),
                    tempFile.toString(),
                    pathManager.getDataJsonPath(lclid)

            );
            processBuilder.start();
            return tempFile.toString();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Adds a performance result to the internal list.
     *
     * @param fogModelPerformance  the performance of the fog model.
     * @param edgeModelPerformance the performance of the edge model.
     * @param date                 the date of the performance result.
     */
    public void addPerformanceResult(Double fogModelPerformance, Double edgeModelPerformance, String date) {
        this.performanceResultList.add(new PerformanceResult(edgeModelPerformance, fogModelPerformance, date));
    }

    /**
     * Saves the performance results to a JSON file.
     */
    public void savePerformanceToJson() {
        Gson gson = new Gson();

        try (FileWriter performanceWriter = new FileWriter(PERFORMANCE_FILE)) {
            gson.toJson(performanceResultList, performanceWriter);
            System.out.println("Successfully saved to json performance result list.");
        } catch (IOException e) {
            System.out.println("Error saving to json the performance result list.");
            throw new RuntimeException(e);
        }
    }

    /**
     * Loads performance results from a JSON file.
     */
    public void loadPerformanceFromJson() {
        Gson gson = new Gson();

        try (FileReader performanceReader = new FileReader(PERFORMANCE_FILE)) {
            Type performanceType = new TypeToken<List<PerformanceResult>>() {}.getType();
            List<PerformanceResult> performanceResultListFromJson = gson.fromJson(performanceReader, performanceType);
            performanceResultList.clear();
            performanceResultList.addAll(performanceResultListFromJson);
            System.out.println("Successfully loaded performance result from json.");
        } catch (IOException e) {
            System.out.println("Not yet loading performance result from json.");
        }
    }
}