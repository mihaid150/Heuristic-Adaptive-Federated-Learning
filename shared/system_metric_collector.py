import threading
import time

import numpy as np
import psutil


class SystemMetricCollector:
    def __init__(self, sampling_rate=2.0):
        """
        Initialize a system metrics collector that samples metrics at fixed interval.
        :param sampling_rate: Sampling rate in seconds.
        """
        self.sampling_rate = sampling_rate
        self.metrics = []
        self.running = False
        self.thread = None

    def _collect_metrics(self):
        """
        Continuously collects system metrics at a fixed interval.
        Additional metrics integrated:
          - CPU frequency (normalized as a percentage relative to the maximum frequency)
          - Sensor temperatures (averaged over all sensors)
        A weighted sum is computed to represent a combined resource load.
        """
        while self.running:
            # Primary metrics
            cpu_usage = psutil.cpu_percent(interval=None)  # non-blocking CPU usage percentage
            memory_usage = psutil.virtual_memory().percent
            load_avg = sum(psutil.getloadavg()) / 3  # average load over 1, 5, and 15 minutes

            # Additional metric: CPU frequency (normalized)
            cpu_freq_obj = psutil.cpu_freq()
            if cpu_freq_obj is not None and cpu_freq_obj.max:
                normalized_cpu_freq = (cpu_freq_obj.current / cpu_freq_obj.max) * 100
            else:
                normalized_cpu_freq = 0

            # Additional metric: Average sensor temperature
            sensor_temps = psutil.sensors_temperatures(fahrenheit=False)
            all_temps = []
            for sensor_list in sensor_temps.values():
                for reading in sensor_list:
                    all_temps.append(reading.current)
            if all_temps:
                average_sensor_temp = sum(all_temps) / len(all_temps)
            else:
                average_sensor_temp = 0

            # cpu_usage: 25%, memory_usage: 25%, load_avg: 20%, normalized CPU frequency: 15%, average sensor temp: 15%
            weights = {
                "cpu_usage": 0.25,
                "memory_usage": 0.25,
                "load_avg": 0.20,
                "normalized_cpu_freq": 0.15,
                "average_sensor_temp": 0.15,
            }

            # Compute the weighted resource load.
            weighted_resource_load = (
                    weights["cpu_usage"] * cpu_usage +
                    weights["memory_usage"] * memory_usage +
                    weights["load_avg"] * load_avg +
                    weights["normalized_cpu_freq"] * normalized_cpu_freq +
                    weights["average_sensor_temp"] * average_sensor_temp
            )

            # Construct the metrics dictionary with all collected values.
            metrics = {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "load_avg": load_avg,
                "normalized_cpu_freq": normalized_cpu_freq,
                "average_sensor_temp": average_sensor_temp,
                "resource_load": weighted_resource_load
            }
            self.metrics.append(metrics)
            time.sleep(self.sampling_rate)

    def start(self):
        """
        Start the metrics collection in a separate thread
        """
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._collect_metrics, daemon=True)
            self.metrics = []
            self.thread.start()

    def stop(self):
        """
        Stop collecting metrics and wait for the thread to terminate
        """
        self.running = False
        if self.thread:
            self.thread.join()

    def get_average_metrics(self):
        """
        Compute the average metrics collected during training.
        """
        if not self.metrics:
            return {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "load_avg": 0.0,
                "normalized_cpu_freq": 0.0,
                "average_sensor_temp": 0.0,
                "resource_load": 0.0
            }
        avg_metrics = {key: np.mean([metric[key] for metric in self.metrics]) for key in self.metrics[0]}
        return avg_metrics
